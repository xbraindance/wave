#include "Cem3387.h"

#include <cmath>
#include <mutex>
#include <unordered_map>

namespace wave::dsp
{
std::shared_ptr<const Cem3387::CutoffCoefficientTable>
Cem3387::coefficientTableForSampleRate(double targetSampleRate)
{
    static std::mutex cacheMutex;
    static std::unordered_map<int, std::weak_ptr<const CutoffCoefficientTable>> cache;
    const auto key = juce::roundToInt(targetSampleRate);
    const std::scoped_lock lock(cacheMutex);
    if (const auto found = cache.find(key); found != cache.end())
        if (auto existing = found->second.lock())
            return existing;

    auto table = std::make_shared<CutoffCoefficientTable>();
    constexpr auto minimumCv = -0.125f;
    constexpr auto maximumCv = 1.125f;
    constexpr auto cutoffRangeOctaves = 127.0f / 12.0f;
    for (size_t index = 0; index < table->size(); ++index)
    {
        const auto unit = static_cast<float>(index)
                          / static_cast<float>(table->size() - 1);
        const auto cv = minimumCv + unit * (maximumCv - minimumCv);
        // Wave voice-card oscillation measurements: 28 Hz, 957 Hz and
        // 7779 Hz at Sound cutoff 0, 62 and 100. Interpolate in log frequency
        // at the analogue control-law boundary. A constant scale previously
        // concealed digital feedback delay at high cutoff. Above the last
        // measured point, retain its scale; that range still needs captures.
        const auto step = cv * 127.0f;
        const auto scaleAt62 = 957.0f / (20.0f * std::exp2(62.0f / 12.0f));
        const auto scaleAt100 = 7779.0f / (20.0f * std::exp2(100.0f / 12.0f));
        const auto logScale = step <= 62.0f
            ? std::log2(1.4f) + juce::jlimit(0.0f, 1.0f, step / 62.0f)
                * (std::log2(scaleAt62) - std::log2(1.4f))
            : std::log2(scaleAt62) + juce::jlimit(0.0f, 1.0f, (step - 62.0f) / 38.0f)
                * (std::log2(scaleAt100) - std::log2(scaleAt62));
        const auto cutoff = juce::jlimit(
            12.0f, static_cast<float>(targetSampleRate * 0.225),
            20.0f * std::exp2(cv * cutoffRangeOctaves + logScale));
        const auto g = std::tan(juce::MathConstants<float>::pi * cutoff
                                / static_cast<float>(targetSampleRate));
        (*table)[index] = g / (1.0f + g);
    }
    cache[key] = table;
    return table;
}

void Cem3387::prepare(double hostSampleRate, float voiceTolerance) noexcept
{
    sampleRate = juce::jmax(1.0, hostSampleRate * 2.0);
    cutoffCoefficientTable = coefficientTableForSampleRate(sampleRate);
    tolerance = juce::jlimit(-1.0f, 1.0f, voiceTolerance);
    cutoffCalibrationManuallySet = false;
    cutoffTrimCode = 0x0800u;
    couplingCoefficient = std::exp(-juce::MathConstants<float>::twoPi * 5.0f
                                   / static_cast<float>(sampleRate * 0.5));
    reset();
}

void Cem3387::reset() noexcept
{
    integrators.fill(0.0f);
    previousInput = 0.0f;
    couplingInput = 0.0f;
    couplingOutput = 0.0f;
    targetCutoffCv = cutoffCv = 0.0f;
    targetResonanceCv = resonanceCv = 0.0f;
    targetPanCv = panCv = 0.5f;
    targetVcaCv = vcaCv = 0.0f;
    resonanceInputGain = 1.0f;
    leftPanGain = rightPanGain = 0.70710678f;
    lastCutoffInput = lastResonanceInput = lastDriveInput = lastAgeInput = -1.0f;
    lastPanInput = -2.0f;
    lastVcaInput = -1.0f;
    coefficientCutoffCv = coefficientResonanceCv = coefficientPanCv = -1.0f;
    controlsInitialised = false;
}

void Cem3387::setControls(float cutoffHz, float resonance, float driveDb, float pan,
                          float circuitAge) noexcept
{
    const auto limitedAge = juce::jlimit(0.0f, 1.0f, circuitAge);
    if (limitedAge != lastAgeInput)
    {
        age = lastAgeInput = limitedAge;
        constexpr auto settlingTimeSeconds = 0.00028f;
        const auto hostSampleRate = static_cast<float>(sampleRate * 0.5);
        const auto timeConstant = settlingTimeSeconds * (1.0f + age * 0.15f);
        controlSlew = 1.0f - std::exp(-1.0f / (hostSampleRate * timeConstant));
        constexpr auto cutoffRangeOctaves = 127.0f / 12.0f;
        const auto toleranceFactor = 1.0f + tolerance * (0.008f + age * 0.035f);
        uncalibratedCutoffOffset
            = std::log2(juce::jmax(0.5f, toleranceFactor)) / cutoffRangeOctaves;
        if (cutoffCalibrationManuallySet)
            applyCutoffCalibrationCode(cutoffTrimCode, true);
        else
            calibrateCutoff();
        coefficientCutoffCv = coefficientResonanceCv = -1.0f;
    }

    constexpr auto cutoffRangeOctaves = 127.0f / 12.0f;
    if (cutoffHz != lastCutoffInput)
    {
        lastCutoffInput = cutoffHz;
        const auto cutoffNormalised
            = std::log2(juce::jlimit(20.0f, 32000.0f, cutoffHz) / 20.0f)
              / cutoffRangeOctaves;
        targetCutoffCv = quantiseCv(cutoffNormalised);
    }
    if (resonance != lastResonanceInput)
    {
        lastResonanceInput = resonance;
        targetResonanceCv = quantiseCv(juce::jlimit(0.0f, 1.0f, resonance));
    }
    if (driveDb != lastDriveInput)
    {
        lastDriveInput = driveDb;
        inputDrive = std::pow(10.0f, juce::jlimit(0.0f, 18.0f, driveDb) / 20.0f);
    }
    if (pan != lastPanInput)
    {
        lastPanInput = pan;
        targetPanCv = quantiseCv((juce::jlimit(-1.0f, 1.0f, pan) + 1.0f) * 0.5f);
    }

    // Static patch controls have already settled before a note sounds. Only
    // subsequent CV writes should exhibit acquisition transients.
    if (!controlsInitialised)
    {
        cutoffCv = targetCutoffCv;
        resonanceCv = targetResonanceCv;
        panCv = targetPanCv;
        controlsInitialised = true;
    }
}

void Cem3387::calibrateCutoff() noexcept
{
    // OS 1.700's VCF service page stores one 12-bit word per voice at
    // $15A680 and masks it with $0FFF. The service monitor displays that word
    // relative to $0800, confirming a bipolar trim centred at mid-scale.
    // Quantise the correction exactly as that trim DAC would: component spread
    // is cancelled, but its sub-LSB residual remains in the analogue path.
    constexpr auto centre = 2048;
    constexpr auto maximum = 4095.0f;
    const auto code = juce::jlimit(
        0, static_cast<int>(maximum),
        juce::roundToInt(static_cast<float>(centre)
                         - uncalibratedCutoffOffset * maximum));
    applyCutoffCalibrationCode(static_cast<uint16_t>(code), true);
}

void Cem3387::setCutoffCalibrationCode(uint16_t code) noexcept
{
    cutoffCalibrationManuallySet = true;
    applyCutoffCalibrationCode(static_cast<uint16_t>(code & 0x0fffu));
}

void Cem3387::applyCutoffCalibrationCode(uint16_t code, bool force) noexcept
{
    constexpr auto centre = 2048;
    constexpr auto maximum = 4095.0f;
    if (!force && cutoffTrimCode == code && cutoffCalibrationManuallySet)
        return;
    cutoffTrimCode = code;
    const auto trimCv = static_cast<float>(static_cast<int>(code) - centre) / maximum;
    cutoffCvOffset = uncalibratedCutoffOffset + trimCv;
    coefficientCutoffCv = -1.0f;
}

Cem3387::StereoSample Cem3387::process(float input, float vcaLevel) noexcept
{
    updateControlVoltages(vcaLevel);

    noiseState ^= noiseState << 13u;
    noiseState ^= noiseState >> 17u;
    noiseState ^= noiseState << 5u;
    const auto noise = static_cast<float>(static_cast<int32_t>(noiseState))
                       / 2147483648.0f;
    // The real feedback loop is never mathematically silent. Thermal/device
    // noise inside the CEM filter supplies the excitation from which maximum
    // resonance reaches sustained oscillation.
    const auto filterExcitation = noise * (0.00000012f + age * 0.0000003f);

    // Two circuit steps per host sample keep the nonlinear four-pole loop stable.
    const auto midpoint = 0.5f * (previousInput + input);
    auto filtered = runFilter(midpoint + filterExcitation);
    filtered = 0.5f * (filtered + runFilter(input + filterExcitation));
    previousInput = input;

    const auto vcaCurvature = vcaCv * vcaCv * (3.0f - 2.0f * vcaCv);
    const auto componentNoise = noise * (0.0000025f + age * 0.000008f);
    const auto vcaBleed = 0.000018f * (1.0f + age * 2.0f);
    const auto rawOutput = std::tanh(filtered * (1.0f + age * 0.08f))
                           * (vcaCurvature + vcaBleed) + componentNoise;
    const auto output = rawOutput - couplingInput
                        + couplingCoefficient * couplingOutput;
    couplingInput = rawOutput;
    couplingOutput = output;

    return { output * leftPanGain, output * rightPanGain };
}

float Cem3387::saturateVcfInput(float input) noexcept
{
    // Waldorf specifies the original input circuit as beginning to saturate
    // at about 70% of maximum oscillator mixer output (a level setting near
    // 75). A normalised tanh with modest drive is essentially unity for small
    // signals, is about 8% compressed at that boundary, and remains smooth
    // beyond it. Do not normalise its full-scale result back to one: that
    // would add makeup gain which is absent from the passive input network.
    constexpr auto inputDriveAtKnee = 0.75f;
    return std::tanh(input * inputDriveAtKnee) / inputDriveAtKnee;
}

void Cem3387::updateControlVoltages(float vcaLevel) noexcept
{
    if (vcaLevel != lastVcaInput)
    {
        lastVcaInput = vcaLevel;
        targetVcaCv = quantiseCv(juce::jlimit(0.0f, 1.0f, vcaLevel));
    }

    // WDV schematic sheets 6-15: one shared AD7545 (CVDA) feeds a PD508
    // multiplexer per voice, 33 nF hold capacitors and TL064 buffers. The
    // schematic RC is only ~15-30 us; this value is a looser, unmeasured
    // choice that softens ideal edges while allowing more of the 53 Hz
    // control stepping through. Age slightly increases acquisition time through switch
    // and capacitor leakage/tolerance.
    const auto settle = [this](float current, float target) {
        return current + (target - current) * controlSlew;
    };

    cutoffCv = settle(cutoffCv, targetCutoffCv);
    resonanceCv = settle(resonanceCv, targetResonanceCv);
    panCv = settle(panCv, targetPanCv);
    vcaCv = settle(vcaCv, targetVcaCv);

    if (cutoffCv != coefficientCutoffCv)
    {
        coefficientCutoffCv = cutoffCv;
        constexpr auto minimumCv = -0.125f;
        constexpr auto maximumCv = 1.125f;
        const auto normalised = juce::jlimit(
            0.0f, 1.0f,
            (cutoffCv + cutoffCvOffset - minimumCv) / (maximumCv - minimumCv));
        const auto position = normalised
                              * static_cast<float>(cutoffCoefficientTable->size() - 1);
        const auto lower = static_cast<size_t>(position);
        const auto upper = juce::jmin(cutoffCoefficientTable->size() - 1, lower + 1);
        const auto fraction = position - static_cast<float>(lower);
        coefficient = (*cutoffCoefficientTable)[lower]
                      + ((*cutoffCoefficientTable)[upper]
                         - (*cutoffCoefficientTable)[lower]) * fraction;
    }
    if (resonanceCv != coefficientResonanceCv)
    {
        coefficientResonanceCv = resonanceCv;
        // Cutoff is service-calibrated per voice. There is no corresponding
        // measured Wave resonance-spread table, so do not add an unrelated
        // synthetic voice signature after calibration.
        resonanceAmount = juce::jlimit(0.0f, 1.04f, resonanceCv);

        // Hardware sine measurements at an open cutoff show passband losses
        // of about 3.8, 5.2, 5.9 and 6.4 dB at resonance 30, 62, 100 and 127.
        // A bare four-pole feedback loop loses over 14 dB at maximum. Model
        // the CEM/input network's makeup gain without changing the autonomous
        // feedback loop, so self-oscillation threshold and level remain genuine.
        constexpr auto feedbackGain = 4.15f;
        constexpr auto maximumPassbandLossDb = 6.4f;
        constexpr auto lossCurve = 3.5f;
        const auto lossPosition
            = (1.0f - std::exp(-lossCurve * resonanceCv))
              / (1.0f - std::exp(-lossCurve));
        const auto measuredPassbandGain = std::pow(
            10.0f, -maximumPassbandLossDb * lossPosition / 20.0f);
        resonanceInputGain
            = (1.0f + feedbackGain * resonanceAmount)
              * measuredPassbandGain;
    }
    if (panCv != coefficientPanCv)
    {
        coefficientPanCv = panCv;
        panPosition = panCv * 2.0f - 1.0f;
        const auto angle = (panPosition + 1.0f)
                           * juce::MathConstants<float>::pi * 0.25f;
        leftPanGain = std::cos(angle);
        rightPanGain = std::sin(angle);
    }
}

float Cem3387::runFilter(float input) noexcept
{
    // Datasheet pp. 5-6: with Ca = 4 Cb the two second-order
    // sections have H(s) = 1 / (1 + s/wc)^4. Four trapezoidal
    // one-poles give that equivalent small-signal response. Close the
    // resonance loop around their *current outputs*, not their stored
    // integrator states: the latter adds artificial feedback phase delay.
    const auto drivenInput = saturateVcfInput(input) * resonanceInputGain * inputDrive;
    const auto feedback = resonanceAmount * 4.15f;
    std::array<float, 4> outputs{};
    const auto evaluate = [&](float stageInput) {
        auto derivative = 1.0f;
        for (size_t stage = 0; stage < outputs.size(); ++stage)
        {
            outputs[stage] = integrators[stage]
                             + coefficient * (stageInput - integrators[stage]);
            derivative *= coefficient;
            if (stage + 1 < outputs.size())
            {
                stageInput = std::tanh(outputs[stage]);
                derivative *= 1.0f - stageInput * stageInput;
            }
        }
        return derivative;
    };

    // The scalar feedback equation is monotone, with derivative >= 1.
    // A safeguarded Newton solve handles saturation without updating any
    // state until the whole loop has been evaluated. Drive acts on the
    // incoming signal only; multiplying feedback by it changes the chip's Q.
    auto stageInput = std::tanh(drivenInput - feedback * integrators[3]);
    if (feedback == 0.0f)
    {
        // With no feedback the first evaluation is already the solution.
        // Keep the same stage arithmetic and saturation as the general solver.
        (void) evaluate(stageInput);
        for (size_t stage = 0; stage < outputs.size(); ++stage)
            integrators[stage] = 2.0f * outputs[stage] - integrators[stage];
        return outputs[3];
    }
    auto lower = -1.0f;
    auto upper = 1.0f;
    for (int iteration = 0; iteration < 8; ++iteration)
    {
        const auto derivative = evaluate(stageInput);
        const auto feedbackInput = std::tanh(drivenInput - feedback * outputs[3]);
        const auto residual = stageInput - feedbackInput;
        if (std::abs(residual) < 1.0e-7f * std::abs(stageInput) + 1.0e-12f
            || iteration == 7)
            break;
        if (residual < 0.0f)
            lower = stageInput;
        else
            upper = stageInput;
        const auto next = stageInput - residual
            / (1.0f + feedback * (1.0f - feedbackInput * feedbackInput) * derivative);
        if (next == stageInput)
            break;
        stageInput = next > lower && next < upper ? next : 0.5f * (lower + upper);
    }
    for (size_t stage = 0; stage < outputs.size(); ++stage)
        integrators[stage] = 2.0f * outputs[stage] - integrators[stage];
    return outputs[3];
}

float Cem3387::quantiseCv(float normalised) noexcept
{
    constexpr auto maximum = 4095.0f; // AD7545 is a 12-bit multiplying DAC.
    return std::round(juce::jlimit(0.0f, 1.0f, normalised) * maximum) / maximum;
}
} // namespace wave::dsp
