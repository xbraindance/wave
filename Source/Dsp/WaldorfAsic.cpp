#include "WaldorfAsic.h"
#if WAVE_HAS_PRIVATE_UPPER_TABLES
#include "FactoryUpperWavetables.h"
#endif
#include "PpgWaveRom.h"
#include <PpgData.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace wave::dsp
{
namespace
{
constexpr double twoPi = juce::MathConstants<double>::twoPi;
constexpr size_t waveSetUserTableOffset = 0x4387c;
constexpr size_t waveSetUserTableRecordBytes = 138;
constexpr size_t waveSetUserWaveOffset = waveSetUserTableOffset
                                         + static_cast<size_t>(WavetableBank::userTableCount)
                                               * waveSetUserTableRecordBytes;
constexpr size_t waveSetUserWaveCount = 1000;
constexpr size_t waveSetHalfWaveSamples = 64;
constexpr size_t minimumWaveSetBytes = waveSetUserWaveOffset
                                       + waveSetUserWaveCount * waveSetHalfWaveSamples;

#if WAVE_HAS_PRIVATE_UPPER_TABLES
template <typename Storage>
bool installUpperFactoryWavetables(Storage& samples)
{
    juce::MemoryOutputStream compressed;
    for (const auto encodedChunk : factory_wavetables::upperFactoryBankZlibBase64)
        if (!juce::Base64::convertFromBase64(
                compressed,
                juce::String::fromUTF8(encodedChunk.data(),
                                       static_cast<int>(encodedChunk.size()))))
            return false;

    juce::MemoryInputStream compressedInput(
        compressed.getData(), compressed.getDataSize(), false);
    juce::GZIPDecompressorInputStream decompressor(
        &compressedInput, false,
        juce::GZIPDecompressorInputStream::zlibFormat,
        static_cast<juce::int64>(factory_wavetables::upperFactoryDecodedBytes));
    std::vector<uint8_t> decoded(factory_wavetables::upperFactoryDecodedBytes);
    size_t bytesRead = 0;
    while (bytesRead < decoded.size())
    {
        const auto amount = decompressor.read(
            decoded.data() + bytesRead,
            static_cast<int>(decoded.size() - bytesRead));
        if (amount <= 0)
            return false;
        bytesRead += static_cast<size_t>(amount);
    }

    constexpr auto firstTable = 30; // Wave display table 31.
    const auto bankIndex = [](int table, int wave, int sample) {
        return (static_cast<size_t>(table) * WavetableBank::wavesPerTable
                + static_cast<size_t>(wave))
               * WavetableBank::samplesPerWave + static_cast<size_t>(sample);
    };
    for (int tableOffset = 0;
         tableOffset < static_cast<int>(factory_wavetables::upperFactoryTableCount);
         ++tableOffset)
    {
        const auto table = firstTable + tableOffset;
        for (int wave = 0;
             wave < static_cast<int>(factory_wavetables::generatedWavesPerTable);
             ++wave)
        {
            for (int sample = 0; sample < 64; ++sample)
            {
                const auto source = (static_cast<size_t>(tableOffset)
                                         * factory_wavetables::generatedWavesPerTable
                                     + static_cast<size_t>(wave))
                                        * factory_wavetables::halfWaveSamples
                                    + static_cast<size_t>(sample);
                const auto value = decoded[source];
                samples[bankIndex(table, wave, sample)]
                    = static_cast<int8_t>(static_cast<int>(value) - 128);
                samples[bankIndex(table, wave, 127 - sample)]
                    = static_cast<int8_t>(
                        static_cast<int>(static_cast<uint8_t>(~value)) - 128);
            }
        }

        // Generated Wave tables use all 61 evolving positions. Their final
        // slots select the hardware's triangle, pulse and square standards.
        uint8_t triangle = 0x20u;
        for (int sample = 0; sample < 64; ++sample)
        {
            samples[bankIndex(table, 61, sample)]
                = static_cast<int8_t>(static_cast<int>(triangle) - 128);
            triangle = static_cast<uint8_t>(triangle + 3u);
        }
        for (int sample = 64; sample < 128; ++sample)
        {
            samples[bankIndex(table, 61, sample)]
                = static_cast<int8_t>(static_cast<int>(triangle) - 128);
            triangle = static_cast<uint8_t>(triangle - 3u);
        }
        for (int sample = 0; sample < 126; ++sample)
            samples[bankIndex(table, 62, sample)] = -48;
        samples[bankIndex(table, 62, 126)] = 127;
        samples[bankIndex(table, 62, 127)] = 127;
        for (int sample = 0; sample < 64; ++sample)
            samples[bankIndex(table, 63, sample)] = -96;
        for (int sample = 64; sample < 128; ++sample)
            samples[bankIndex(table, 63, sample)] = 96;
    }
    return true;
}

#endif

double spectralAmplitude(int family, int harmonic, double brightness)
{
    const auto h = static_cast<double>(harmonic);
    switch (family % 8)
    {
        case 0: return std::pow(brightness, h * 0.42) / h;
        case 1: return (harmonic % 2 != 0) ? std::pow(brightness, h * 0.34) / h : 0.0;
        case 2: return (harmonic % 3 != 0) ? std::pow(brightness, h * 0.28) / h : 0.0;
        case 3: return std::pow(brightness, h * 0.5) / (h * h * 0.16 + 1.0);
        case 4: return (harmonic == 1 || harmonic == 2 || harmonic == 5 || harmonic == 9)
                           ? std::pow(brightness, h * 0.18) / std::sqrt(h)
                           : 0.0;
        case 5: return (harmonic % 4 == 1) ? std::pow(brightness, h * 0.24) / std::sqrt(h) : 0.0;
        case 6: return std::sin(h * 1.61803398875) * std::pow(brightness, h * 0.3) / std::sqrt(h);
        default: return std::cos(h * 0.73) * std::pow(brightness, h * 0.37) / h;
    }
}
} // namespace

WavetableBank::WavetableBank()
{
    static const auto sharedFallback = [] {
        auto generated = std::make_shared<SampleStorage>();
        for (int table = 0; table < numTables; ++table)
        {
            const auto family = table % 8;
            const auto phaseSkew = static_cast<double>((table * 37) % 127) / 127.0;

            for (int wave = 0; wave < wavesPerTable; ++wave)
            {
                const auto brightness = 0.06 + 0.93 * static_cast<double>(wave)
                                                       / static_cast<double>(wavesPerTable - 1);
                std::array<double, samplesPerWave> floating{};
                double peak = 0.000001;

                for (int i = 0; i < samplesPerWave; ++i)
                {
                    const auto p = static_cast<double>(i) / samplesPerWave;
                    double value = 0.0;
                    for (int harmonic = 1; harmonic <= 48; ++harmonic)
                    {
                        const auto amplitude = spectralAmplitude(family, harmonic, brightness);
                        const auto harmonicPhase
                            = family >= 6 ? phaseSkew * harmonic * 0.31 : 0.0;
                        value += amplitude * std::sin(
                            twoPi * (p * harmonic + harmonicPhase));
                    }
                    floating[static_cast<size_t>(i)] = value;
                    peak = std::max(peak, std::abs(value));
                }

                for (int i = 0; i < samplesPerWave; ++i)
                {
                    const auto normalised = floating[static_cast<size_t>(i)] / peak;
                    const auto quantised = juce::jlimit(
                        -127, 127, juce::roundToInt(normalised * 127.0));
                    (*generated)[index(table, wave, i)]
                        = static_cast<int8_t>(quantised);
                }
            }
        }

        constexpr auto ppgSampleCount = PpgWaveRom::decodedTableCount
                                         * wavesPerTable * samplesPerWave;
        jassert(WavePpgAssets::ppgv6wavetables_binSize == ppgSampleCount);
        std::memcpy(generated->data(), WavePpgAssets::ppgv6wavetables_bin,
                    static_cast<size_t>(ppgSampleCount));
#if WAVE_HAS_PRIVATE_UPPER_TABLES
        const auto installed = installUpperFactoryWavetables(*generated);
        jassert(installed);
        juce::ignoreUnused(installed);
#endif
        return generated;
    }();
    samples = sharedFallback;
}

void WavetableBank::detachSamples()
{
    if (samples.use_count() > 1)
        samples = std::make_shared<SampleStorage>(*samples);
}

bool WavetableBank::loadSigned8BitRom(const void* data, size_t size) noexcept
{
    constexpr auto factorySamples = static_cast<size_t>(factoryTableCount)
                                    * wavesPerTable * samplesPerWave;
    if (data == nullptr || (size != samples->size() && size != factorySamples))
        return false;
    detachSamples();
    std::memcpy(samples->data(), data, size);
    externalRomLoaded = true;
    importedTables = size == samples->size() ? numTables : factoryTableCount;
    source = size == samples->size()
                 ? "User-supplied 128-table signed 8-bit image"
                 : "User-supplied 64-table signed 8-bit image";
    return true;
}

bool WavetableBank::loadFirst32Signed8BitRom(const void* data, size_t size) noexcept
{
    constexpr auto first32Samples = static_cast<size_t>(32) * wavesPerTable * samplesPerWave;
    if (data == nullptr || size != first32Samples)
        return false;
    detachSamples();
    std::memcpy(samples->data(), data, first32Samples);
    externalRomLoaded = true;
    importedTables = 32;
    source = "User-supplied first-32-table signed 8-bit image";
    return true;
}

bool WavetableBank::loadPpgWaveRom(const void* data, size_t size) noexcept
{
    try
    {
        constexpr auto decodedSamples = static_cast<size_t>(PpgWaveRom::decodedTableCount)
                                        * wavesPerTable * samplesPerWave;
        std::vector<int8_t> decoded(decodedSamples);
        const auto result = PpgWaveRom::decode(data, size, decoded);
        if (!result.success)
            return false;

        detachSamples();
        std::copy(decoded.begin(), decoded.end(), samples->begin());
        externalRomLoaded = true;
        importedTables = result.decodedTables;
        source = "User-supplied PPG Wave 2.3 V6 EPROM (30 lower tables reconstructed)";
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool WavetableBank::loadRomImage(const void* data, size_t size) noexcept
{
    constexpr auto factorySamples = static_cast<size_t>(factoryTableCount)
                                    * wavesPerTable * samplesPerWave;
    constexpr auto first32Samples = static_cast<size_t>(32) * wavesPerTable * samplesPerWave;
    if (size == samples->size() || size == factorySamples)
        return loadSigned8BitRom(data, size);
    if (size == first32Samples)
        return loadFirst32Signed8BitRom(data, size);
    return loadPpgWaveRom(data, size);
}

bool WavetableBank::loadWaveSetUserTables(const void* data, size_t size) noexcept
{
    if (data == nullptr || size < minimumWaveSetBytes)
        return false;

    const auto* bytes = static_cast<const uint8_t*>(data);
    for (int table = 0; table < userTableCount; ++table)
    {
        const auto record = waveSetUserTableOffset
                            + static_cast<size_t>(table) * waveSetUserTableRecordBytes;
        if (bytes[record + 9] != 0x55u)
            return false;
    }

    detachSamples();
    for (int userTable = 0; userTable < userTableCount; ++userTable)
    {
        const auto destinationTable = factoryTableCount + userTable;
        const auto record = waveSetUserTableOffset
                            + static_cast<size_t>(userTable) * waveSetUserTableRecordBytes;
        std::array<bool, wavesPerTable> populated{};

        for (int wave = 0; wave < wavesPerTable; ++wave)
        {
            const auto referenceOffset = record + 10 + static_cast<size_t>(wave) * 2;
            const auto reference = static_cast<uint16_t>(
                (static_cast<uint16_t>(bytes[referenceOffset]) << 8u)
                | bytes[referenceOffset + 1]);
            if (reference == 0xffffu)
                continue;

            if (reference < 0x1000u)
            {
                // Factory Wave references are a flat 0..4095 Wave number,
                // not an encoded table/position byte pair.
                const auto sourceTable = static_cast<int>(reference) / wavesPerTable;
                const auto sourceWave = static_cast<int>(reference) % wavesPerTable;
                if (sourceTable >= factoryTableCount)
                    return false;
                for (int sampleIndex = 0; sampleIndex < samplesPerWave; ++sampleIndex)
                    (*samples)[index(destinationTable, wave, sampleIndex)]
                        = (*samples)[index(sourceTable, sourceWave, sampleIndex)];
            }
            else if (reference < 0x2000u)
            {
                const auto userWave = static_cast<size_t>(reference & 0x0fffu);
                if (userWave >= waveSetUserWaveCount)
                    return false;
                const auto* halfWave = bytes + waveSetUserWaveOffset
                                       + userWave * waveSetHalfWaveSamples;
                for (int sampleIndex = 0; sampleIndex < 64; ++sampleIndex)
                {
                    const auto value = halfWave[static_cast<size_t>(sampleIndex)];
                    (*samples)[index(destinationTable, wave, sampleIndex)]
                        = static_cast<int8_t>(static_cast<int>(value) - 128);
                    (*samples)[index(destinationTable, wave, 127 - sampleIndex)]
                        = static_cast<int8_t>(static_cast<int>(
                                                  static_cast<uint8_t>(~value))
                                              - 128);
                }
            }
            else
            {
                return false;
            }
            populated[static_cast<size_t>(wave)] = true;
        }

        // Empty tail positions have the Wave's standard pulse, square and
        // saw waves. Explicit user choices still take precedence.
        for (int wave = 61; wave < wavesPerTable; ++wave)
        {
            if (populated[static_cast<size_t>(wave)])
                continue;
            for (int sampleIndex = 0; sampleIndex < samplesPerWave; ++sampleIndex)
                (*samples)[index(destinationTable, wave, sampleIndex)]
                    = (*samples)[index(0, wave, sampleIndex)];
            populated[static_cast<size_t>(wave)] = true;
        }

        // Wave OS repeatedly creates the midpoint of each remaining gap.
        // This retains its quantised averaging character instead of using a
        // floating-point crossfade across the complete gap.
        for (;;)
        {
            auto complete = true;
            auto left = 0;
            while (left < wavesPerTable)
            {
                while (left < wavesPerTable && !populated[static_cast<size_t>(left)])
                    ++left;
                if (left >= wavesPerTable - 1)
                    break;
                auto right = left + 1;
                while (right < wavesPerTable && !populated[static_cast<size_t>(right)])
                    ++right;
                if (right >= wavesPerTable)
                    break;
                if (right - left > 1)
                {
                    const auto midpoint = left + (right - left) / 2;
                    for (int sampleIndex = 0; sampleIndex < samplesPerWave; ++sampleIndex)
                    {
                        const auto leftUnsigned = static_cast<unsigned>(
                            static_cast<int>((*samples)[index(destinationTable, left,
                                                               sampleIndex)])
                            + 128);
                        const auto rightUnsigned = static_cast<unsigned>(
                            static_cast<int>((*samples)[index(destinationTable, right,
                                                               sampleIndex)])
                            + 128);
                        (*samples)[index(destinationTable, midpoint, sampleIndex)]
                            = static_cast<int8_t>(static_cast<int>(
                                                      (leftUnsigned + rightUnsigned) >> 1u)
                                                  - 128);
                    }
                    populated[static_cast<size_t>(midpoint)] = true;
                    complete = false;
                }
                left = right;
            }
            if (complete)
                break;
        }
    }

    waveSetUserTablesLoaded = true;
    if (!source.contains("Wave SET user tables"))
        source += " + Wave SET user tables";
    return true;
}

float WavetableBank::sample(int table, float position, double phase, bool smoothPosition) const noexcept
{
    return static_cast<float>(sampleCode(table, position, phase, smoothPosition))
           / 128.0f;
}

int8_t WavetableBank::sampleCode(int table, float position, double phase,
                                 bool smoothPosition) const noexcept
{
    table = juce::jlimit(0, numTables - 1, table);
    position = juce::jlimit(0.0f, static_cast<float>(wavesPerTable - 1), position);
    phase -= std::floor(phase);

    const auto lowerWave = juce::jlimit(0, wavesPerTable - 1, static_cast<int>(position));
    const auto upperWave = juce::jmin(wavesPerTable - 1, lowerWave + 1);
    const auto phaseIndex = juce::jlimit(0, samplesPerWave - 1,
                                        static_cast<int>(phase * samplesPerWave));
    const auto lower = static_cast<int>(rawSample(table, lowerWave, phaseIndex));

    if (!smoothPosition || lowerWave == upperWave)
        return static_cast<int8_t>(lower);

    const auto upper = static_cast<int>(rawSample(table, upperWave, phaseIndex));
    return static_cast<int8_t>(juce::jlimit(
        -128, 127,
        juce::roundToInt(static_cast<float>(lower)
                         + static_cast<float>(upper - lower)
                               * (position - static_cast<float>(lowerWave)))));
}

int8_t WavetableBank::rawSample(int table, int position, int sampleIndex) const noexcept
{
    return (*samples)[index(juce::jlimit(0, numTables - 1, table),
                            juce::jlimit(0, wavesPerTable - 1, position),
                            juce::jlimit(0, samplesPerWave - 1, sampleIndex))];
}

size_t WavetableBank::index(int table, int position, int sampleIndex) noexcept
{
    return (static_cast<size_t>(table) * wavesPerTable + static_cast<size_t>(position))
           * samplesPerWave + static_cast<size_t>(sampleIndex);
}

void AsicResampler::prepare(double hostSampleRate)
{
    // Coefficients are shared by all voices and only built during prepare.
    static std::mutex cacheMutex;
    static std::map<double, std::weak_ptr<const Kernel>> cache;
    const auto rate = juce::jmax(1.0, hostSampleRate);
    const std::lock_guard<std::mutex> lock(cacheMutex);
    kernel = cache[rate].lock();
    if (kernel == nullptr)
    {
        auto next = std::make_shared<Kernel>();
        const auto bandwidth = juce::jmin(rate, OscillatorChipProxy::modelClockRate());
        const auto ratio = OscillatorChipProxy::modelClockRate() / bandwidth;
        // Blackman-windowed sinc: flat through 0.40 * bandwidth, with a
        // transition centred at 0.45 and rejection by host Nyquist. Bound
        // storage for pathological non-audio rates; normal audio rates use
        // the full 96-tap-per-output-period design.
        next->taps = 4 * static_cast<int>(std::ceil(juce::jmin(8192.0, 24.0 * ratio)));
        next->coefficients.resize(static_cast<size_t>((phaseCount + 1) * next->taps));
        const auto cutoff = 0.45 / ratio;
        const auto radius = static_cast<double>(next->taps) * 0.5;
        for (int phaseIndex = 0; phaseIndex <= phaseCount; ++phaseIndex)
        {
            const auto fraction = static_cast<double>(phaseIndex) / phaseCount;
            auto sum = 0.0;
            auto* coefficients = next->coefficients.data() + phaseIndex * next->taps;
            for (int tap = 0; tap < next->taps; ++tap)
            {
                const auto distance = static_cast<double>(tap) + fraction - radius;
                const auto angle = twoPi * cutoff * distance;
                const auto sinc = std::abs(angle) < 1.0e-12 ? 1.0 : std::sin(angle) / angle;
                const auto window = 0.42 + 0.5 * std::cos(juce::MathConstants<double>::pi
                                                         * distance / radius)
                                         + 0.08 * std::cos(twoPi * distance / radius);
                coefficients[tap] = static_cast<float>(2.0 * cutoff * sinc * window);
                sum += coefficients[tap];
            }
            for (int tap = 0; tap < next->taps; ++tap)
                coefficients[tap] = static_cast<float>(coefficients[tap] / sum);
        }
        kernel = next;
        cache[rate] = kernel;
    }
    // Mirroring the ring keeps the dot products contiguous without per-tap
    // modulus operations. No allocations or coefficient generation in render.
    history.resize(static_cast<size_t>(2 * kernel->taps));
    reset();
}

void AsicResampler::reset() noexcept
{
    std::fill(history.begin(), history.end(), 0.0f);
    head = 0;
}

void AsicResampler::push(float sample) noexcept
{
    if (--head < 0)
        head = kernel->taps - 1;
    history[static_cast<size_t>(head)] = sample;
    history[static_cast<size_t>(head + kernel->taps)] = sample;
}

float AsicResampler::read(double fractionalTick) const noexcept
{
    const auto phase = juce::jlimit(0.0, 1.0, fractionalTick) * phaseCount;
    const auto index = juce::jmin(phaseCount - 1, static_cast<int>(phase));
    const auto blend = static_cast<float>(phase - index);
    const auto* first = kernel->coefficients.data() + index * kernel->taps;
    const auto* second = first + kernel->taps;
    const auto* samples = history.data() + head;
    // Independent lanes let the compiler vectorise both dot products without
    // enabling fast-math or reassociating a single serial accumulation.
    std::array<float, 4> a {};
    std::array<float, 4> b {};
    for (int tap = 0; tap < kernel->taps; tap += 4)
    {
        for (int lane = 0; lane < 4; ++lane)
        {
            a[static_cast<size_t>(lane)] += samples[tap + lane] * first[tap + lane];
            b[static_cast<size_t>(lane)] += samples[tap + lane] * second[tap + lane];
        }
    }
    const auto firstSum = (a[0] + a[1]) + (a[2] + a[3]);
    const auto secondSum = (b[0] + b[1]) + (b[2] + b[3]);
    return firstSum + blend * (secondSum - firstSum);
}

void OscillatorChipProxy::prepare(double hostSampleRate)
{
    sampleRate = juce::jmax(1.0, hostSampleRate);
    resampler.prepare(sampleRate);
    reset();
}

void OscillatorChipProxy::reset(double startPhase) noexcept
{
    phase = startPhase - std::floor(startPhase);
    clockPhase = 1.0;
    heldSample = 0.0f;
    resampler.reset();
}

void OscillatorChipProxy::setFrequency(float frequencyHz) noexcept
{
    phaseIncrementPerTick = juce::jlimit(0.0, 0.49,
                                         static_cast<double>(frequencyHz) / modelClockRate());
}

float OscillatorChipProxy::process(const WavetableBank& bank, int table, float position,
                                   bool smoothPosition) noexcept
{
    clockPhase += modelClockRate() / sampleRate;
    while (clockPhase >= 1.0)
    {
        clockPhase -= 1.0;
        heldSample = tick(bank, table, position, smoothPosition);
        resampler.push(heldSample);
    }
    return resampler.read(clockPhase);
}

float OscillatorChipProxy::tick(const WavetableBank& bank, int table, float position,
                                bool smoothPosition) noexcept
{
    return static_cast<float>(tickCode(bank, table, position, smoothPosition))
           / 128.0f;
}

int8_t OscillatorChipProxy::tickCode(const WavetableBank& bank, int table,
                                     float position, bool smoothPosition) noexcept
{
    phase += phaseIncrementPerTick;
    phase -= std::floor(phase);
    const auto code = bank.sampleCode(table, position, phase, smoothPosition);
    heldSample = static_cast<float>(code) / 128.0f;
    return code;
}

int8_t AsicOutputMixer::mixOscillatorCodes(int8_t oscillator1,
                                           int8_t oscillator2,
                                           uint8_t level1,
                                           uint8_t level2) noexcept
{
    const auto product = static_cast<int>(oscillator1)
                             * static_cast<int>(level1 & 0x7fu)
                         + static_cast<int>(oscillator2)
                               * static_cast<int>(level2 & 0x7fu);
    // Match an arithmetic shift rather than C++'s truncation toward zero for
    // negative products. Keep the operation explicit and portable.
    const auto scaled = product >= 0 ? product / 128
                                     : -static_cast<int>((-product + 127) / 128);
    const auto wrapped = static_cast<unsigned int>(scaled) & 0xffu;
    return static_cast<int8_t>(wrapped < 128u
                                   ? static_cast<int>(wrapped)
                                   : static_cast<int>(wrapped) - 256);
}

int8_t AsicOutputMixer::quantiseOscillator(float sample) noexcept
{
    return static_cast<int8_t>(juce::jlimit(
        -128, 127, juce::roundToInt(juce::jlimit(-1.0f, 127.0f / 128.0f, sample)
                                    * 128.0f)));
}

float AsicOutputMixer::normaliseOutput(int8_t sample) noexcept
{
    return static_cast<float>(sample) / 128.0f;
}

void AsicHighpassFilter::prepare(double newSampleRate) noexcept
{
    sampleRate = juce::jmax(1.0, newSampleRate);
    lastCutoff = -1.0f;
    reset();
}

void AsicHighpassFilter::reset() noexcept
{
    integrator1 = 0.0f;
    integrator2 = 0.0f;
}

float AsicHighpassFilter::process(float input, float cutoffHz) noexcept
{
    // The Wave documentation exposes a non-resonant 12 dB digital high-pass,
    // but the ASIC arithmetic is undocumented. A topology-preserving TPT
    // two-pole section supplies that measured boundary without pretending to
    // recover unknown gates. The output retains signed 16-bit fixed-point
    // stepping before entering the analogue reconstruction path.
    const auto cutoff = juce::jlimit(10.0f, static_cast<float>(sampleRate * 0.45), cutoffHz);
    constexpr auto damping = 1.41421356237f;
    if (cutoff != lastCutoff)
    {
        lastCutoff = cutoff;
        const auto g = std::tan(juce::MathConstants<float>::pi * cutoff
                                / static_cast<float>(sampleRate));
        a1 = 1.0f / (1.0f + g * (g + damping));
        a2 = g * a1;
        a3 = g * a2;
    }
    const auto v3 = input - integrator2;
    const auto v1 = a1 * integrator1 + a2 * v3;
    const auto v2 = integrator2 + a2 * integrator1 + a3 * v3;
    integrator1 = 2.0f * v1 - integrator1;
    integrator2 = 2.0f * v2 - integrator2;
    const auto highpass = input - damping * v1 - v2;
    return std::round(juce::jlimit(-1.0f, 1.0f, highpass) * 32767.0f) / 32768.0f;
}

void ReconstructionStage::prepare(double newSampleRate, float voiceTolerance) noexcept
{
    sampleRate = juce::jmax(1.0, newSampleRate);
    // The network is built from fixed 1% resistors and capacitors with no
    // service trim, so no per-voice spread is invented for it.
    juce::ignoreUnused(voiceTolerance);
    tolerance = 0.0f;
    coefficientAge = -1.0f;
    updateCoefficients();
    reset();
}

void ReconstructionStage::reset() noexcept
{
    sectionState.fill(0.0f);
    dcState = 0.0f;
}

void ReconstructionStage::setAge(float amount) noexcept
{
    const auto limited = juce::jlimit(0.0f, 1.0f, amount);
    if (limited != age || coefficientAge < 0.0f)
    {
        age = limited;
        updateCoefficients();
    }
}

void ReconstructionStage::updateCoefficients() noexcept
{
    coefficientAge = age;
    // Cache the nonlinear transfer at every DAC code instead of calling tanh
    // at 250 kHz. The saturation itself is not from the schematic.
    for (int code = -dacCodeLimit; code <= dacCodeLimit; ++code)
        saturatedLevels[static_cast<size_t>(code + dacCodeLimit)] = std::tanh(
            (static_cast<float>(code) / static_cast<float>(dacCodeLimit))
            * (1.0f + age * 0.16f));

    // Poles and zeros (Hz) of the WVC input network, ideal op-amp and source
    // (scripts/derive_wvc_input_network.py): the C4003/R4003 zero cancels the
    // 4019 Hz pole, leaving six poles and two zeros. The stage's inversion is not
    // modelled (all voices share it).
    struct Section { double zeroHz; double poleHz; };
    static constexpr std::array<Section, sectionCount> sections {{
        { 25.6536, 10.2542 },    // C4001/R4004 feedback: bass shelf (10 Hz = 1/2piR4004C4001)
        { 1808.579, 3784.817 },  // R4002/R4003/C4003 input vs R4005 feedback: treble shelf
        { 0.0, 21937.57 },       // four upper poles from C4000, C4002, C4004, R4006/C4005
        { 0.0, 23154.02 },
        { 0.0, 29019.30 },
        { 0.0, 41711.73 },
    }};
    const auto k = 2.0 * sampleRate;
    const auto prewarp = [&](double hz) {
        const auto limited = juce::jmin(hz, sampleRate * 0.45);
        return k * std::tan(juce::MathConstants<double>::pi * limited / sampleRate);
    };
    for (size_t index = 0; index < sections.size(); ++index)
    {
        const auto wp = prewarp(sections[index].poleHz);
        const auto wz = sections[index].zeroHz > 0.0 ? prewarp(sections[index].zeroHz) : 0.0;
        // H(s) = (1 + s/wz) / (1 + s/wp), s = k (1 - 1/z) / (1 + 1/z).
        const auto zeroTerm = wz > 0.0 ? k / wz : 0.0;
        const auto poleTerm = k / wp;
        const auto norm = 1.0 / (1.0 + poleTerm);
        b0[index] = static_cast<float>((1.0 + zeroTerm) * norm);
        b1[index] = static_cast<float>((1.0 - zeroTerm) * norm);
        a1[index] = static_cast<float>((1.0 - poleTerm) * norm);
    }
    highPassCoefficient = std::exp(-juce::MathConstants<float>::twoPi * 7.0f
                                   / static_cast<float>(sampleRate));
}

float ReconstructionStage::process(float input) noexcept
{
    // OscillatorChipProxy has already performed the digital sample-and-hold:
    // its output only changes when the emulated oscillator clock advances,
    // so the DAC code drives the analogue network directly.
    const auto code = static_cast<int>(std::round(
        juce::jlimit(-1.0f, 1.0f, input) * static_cast<float>(dacCodeLimit)));
    // DC gain of the network: (R4004 + R4005) / R4002 = 2.5, less the
    // R4000/R4001 and R4006/R4014 dividers. The shelves are relative to it.
    constexpr auto networkDcGain = 2.3381f;
    auto signal = networkDcGain * saturatedLevels[static_cast<size_t>(code + dacCodeLimit)];
    for (size_t index = 0; index < sectionCount; ++index)
    {
        // Transposed direct form II first-order section.
        const auto out = b0[index] * signal + sectionState[index];
        sectionState[index] = b1[index] * signal - a1[index] * out;
        signal = out;
    }

    // C4006 (10 uF) couples the network to SIN. The 7 Hz corner is not from
    // the schematic: it depends on the CEM3387 SIN input impedance.
    const auto dc = signal + highPassCoefficient * (dcState - signal);
    const auto output = signal - dc;
    dcState = dc;
    return output;
}
} // namespace wave::dsp
