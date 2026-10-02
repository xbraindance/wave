#pragma once

#include <juce_core/juce_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace wave::dsp
{
class WavetableBank
{
public:
    static constexpr int factoryTableCount = 64;
    static constexpr int userTableCount = 64;
    static constexpr int numTables = factoryTableCount + userTableCount;
    static constexpr int wavesPerTable = 64;
    static constexpr int samplesPerWave = 128;

    WavetableBank();

    bool loadSigned8BitRom(const void* data, size_t size) noexcept;
    bool loadFirst32Signed8BitRom(const void* data, size_t size) noexcept;
    bool loadPpgWaveRom(const void* data, size_t size) noexcept;
    bool loadRomImage(const void* data, size_t size) noexcept;
    bool loadWaveSetUserTables(const void* data, size_t size) noexcept;
    [[nodiscard]] bool isExternalRomLoaded() const noexcept { return externalRomLoaded; }
    [[nodiscard]] int importedTableCount() const noexcept { return importedTables; }
    [[nodiscard]] bool hasWaveSetUserTables() const noexcept { return waveSetUserTablesLoaded; }
    [[nodiscard]] const juce::String& sourceDescription() const noexcept { return source; }

    [[nodiscard]] float sample(int table, float position, double phase,
                               bool smoothPosition = true) const noexcept;
    [[nodiscard]] int8_t sampleCode(int table, float position, double phase,
                                    bool smoothPosition = true) const noexcept;
    [[nodiscard]] int8_t rawSample(int table, int position, int sampleIndex) const noexcept;

private:
    static constexpr size_t totalSamples = static_cast<size_t>(numTables)
                                           * wavesPerTable * samplesPerWave;
    using SampleStorage = std::array<int8_t, totalSamples>;
    std::shared_ptr<SampleStorage> samples;
    bool externalRomLoaded = false;
    bool waveSetUserTablesLoaded = false;
    int importedTables = 0;
    juce::String source { "PPG V6 lower tables + procedural upper tables" };

    [[nodiscard]] static size_t index(int table, int position, int sampleIndex) noexcept;
    void detachSamples();
};

// Bandlimited conversion from the fixed ASIC clock to the host clock. Feed
// every internal tick, then read at the fractional host sampling instant.
// The causal FIR adds about 1.1 ms at 44.1 kHz; it is not an analogue model.
class AsicResampler
{
public:
    void prepare(double hostSampleRate);
    void reset() noexcept;
    void push(float sample) noexcept;
    [[nodiscard]] float read(double fractionalTick) const noexcept;

private:
    static constexpr int phaseCount = 64;
    struct Kernel
    {
        int taps = 0;
        std::vector<float> coefficients;
    };
    std::shared_ptr<const Kernel> kernel;
    std::vector<float> history;
    int head = 0;
};

// External behavioural proxy for the undocumented oscillator chip. This does
// not claim to reproduce, or know, its internal implementation.
class OscillatorChipProxy
{
public:
    void prepare(double hostSampleRate);
    void reset(double startPhase = 0.0) noexcept;
    void setFrequency(float frequencyHz) noexcept;
    [[nodiscard]] float frequencyHz() const noexcept
    {
        return static_cast<float>(phaseIncrementPerTick * modelClockRate());
    }

    [[nodiscard]] float process(const WavetableBank& bank, int table, float position,
                                bool smoothPosition = true) noexcept;
    [[nodiscard]] float tick(const WavetableBank& bank, int table, float position,
                             bool smoothPosition = true) noexcept;
    [[nodiscard]] int8_t tickCode(const WavetableBank& bank, int table,
                                  float position,
                                  bool smoothPosition = true) noexcept;

    // Waldorf's measured ASIC recreation uses the original chip's internal
    // synthesis domain regardless of the host/DAW sample rate.
    [[nodiscard]] static constexpr double modelClockRate() noexcept { return 250000.0; }

private:
    double sampleRate = 44100.0;
    double phase = 0.0;
    double clockPhase = 1.0;
    double phaseIncrementPerTick = 440.0 / modelClockRate();
    float heldSample = 0.0f;
    AsicResampler resampler;
};

// The ES2 ASIC combines its two signed eight-bit oscillator values after
// multiplying them by unsigned seven-bit level codes. Its output accumulator
// is only eight bits wide: crossing the positive or negative boundary wraps
// with two's-complement polarity inversion instead of clipping. Waldorf later
// exposed this original defect as the "ASIC Mix Bug" on the M.
class AsicOutputMixer
{
public:
    [[nodiscard]] static int8_t mixOscillatorCodes(int8_t oscillator1,
                                                   int8_t oscillator2,
                                                   uint8_t level1,
                                                   uint8_t level2) noexcept;
    // The WDV firmware sends each oscillator level to the ASIC as only three
    // bits (ws level >> 4, OS 1.700 WDV 0x9B8/0xA58). Scaling that code back to
    // the 0..0x70 level range is a hypothesis about the ASIC's gain law, not a
    // documented fact.
    [[nodiscard]] static uint8_t levelFromRegisterBits(uint8_t level) noexcept
    {
        return static_cast<uint8_t>(level & 0x70u);
    }
    [[nodiscard]] static int8_t quantiseOscillator(float sample) noexcept;
    [[nodiscard]] static float normaliseOutput(int8_t sample) noexcept;
};

// Observable 12 dB/octave high-pass boundary of the otherwise undocumented
// Waldorf ASIC. It sits before the analogue reconstruction/CEM3387 path.
class AsicHighpassFilter
{
public:
    void prepare(double sampleRate) noexcept;
    void reset() noexcept;
    [[nodiscard]] float process(float input, float cutoffHz) noexcept;

private:
    double sampleRate = 44100.0;
    float integrator1 = 0.0f;
    float integrator2 = 0.0f;
    float lastCutoff = -1.0f;
    float a1 = 1.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
};

// Voice-card signal path between the per-voice AD7545 DAC and the CEM3387 SIN
// pin: WVC input network (R4000-R4006, C4000-C4005, TL062), derived from the
// service manual schematic by scripts/derive_wvc_input_network.py.
class ReconstructionStage
{
public:
    void prepare(double newSampleRate, float voiceTolerance) noexcept;
    void reset() noexcept;
    void setAge(float amount) noexcept;
    [[nodiscard]] float process(float input) noexcept;

    // The WDV board feeds each voice through a 12-bit AD7545 whose two LSB
    // inputs are tied low (WD0-WD9 drive DB2-DB11): a 10-bit signed code.
    static constexpr int dacCodeLimit = 511;

private:
    void updateCoefficients() noexcept;

    static constexpr size_t sectionCount = 6;
    double sampleRate = 44100.0;
    std::array<float, sectionCount> sectionState{};
    std::array<float, sectionCount> b0{}, b1{}, a1{};
    float dcState = 0.0f;
    float tolerance = 0.0f;
    float age = 0.0f;
    float coefficientAge = -1.0f;
    std::array<float, 2 * dacCodeLimit + 1> saturatedLevels{};
    float highPassCoefficient = 0.0f;
};
} // namespace wave::dsp
