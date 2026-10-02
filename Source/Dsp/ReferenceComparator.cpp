#include "ReferenceComparator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace wave::dsp
{
ReferenceComparator::Metrics ReferenceComparator::compare(
    const juce::AudioBuffer<float>& reference, const juce::AudioBuffer<float>& candidate,
    int maximumLagSamples) noexcept
{
    Metrics result;
    const auto channels = juce::jmin(reference.getNumChannels(), candidate.getNumChannels());
    const auto samples = juce::jmin(reference.getNumSamples(), candidate.getNumSamples());
    if (channels <= 0 || samples <= 0)
        return result;

    maximumLagSamples = juce::jlimit(0, samples - 1, maximumLagSamples);
    const auto lagSearchStride = juce::jmax(1, samples / 4096);
    auto bestScore = -std::numeric_limits<double>::infinity();
    for (int lag = -maximumLagSamples; lag <= maximumLagSamples; ++lag)
    {
        double dot = 0.0;
        double referenceEnergy = 0.0;
        double candidateEnergy = 0.0;
        const auto start = juce::jmax(0, -lag);
        const auto end = juce::jmin(samples, samples - lag);
        for (int channel = 0; channel < channels; ++channel)
        {
            const auto* ref = reference.getReadPointer(channel);
            const auto* model = candidate.getReadPointer(channel);
            for (int sample = start; sample < end; sample += lagSearchStride)
            {
                const auto a = static_cast<double>(ref[sample]);
                const auto b = static_cast<double>(model[sample + lag]);
                dot += a * b;
                referenceEnergy += a * a;
                candidateEnergy += b * b;
            }
        }
        const auto denominator = std::sqrt(referenceEnergy * candidateEnergy);
        const auto score = denominator > 1.0e-18 ? dot / denominator : 0.0;
        if (score > bestScore)
        {
            bestScore = score;
            result.lagSamples = lag;
        }
    }

    double dot = 0.0;
    double referenceEnergy = 0.0;
    double candidateEnergy = 0.0;
    const auto start = juce::jmax(0, -result.lagSamples);
    const auto end = juce::jmin(samples, samples - result.lagSamples);
    for (int channel = 0; channel < channels; ++channel)
    {
        const auto* ref = reference.getReadPointer(channel);
        const auto* model = candidate.getReadPointer(channel);
        for (int sample = start; sample < end; ++sample)
        {
            const auto a = static_cast<double>(ref[sample]);
            const auto b = static_cast<double>(model[sample + result.lagSamples]);
            dot += a * b;
            referenceEnergy += a * a;
            candidateEnergy += b * b;
        }
    }
    result.fittedGain = candidateEnergy > 1.0e-18
                            ? static_cast<float>(dot / candidateEnergy)
                            : 1.0f;
    result.correlation = referenceEnergy > 1.0e-18 && candidateEnergy > 1.0e-18
                             ? static_cast<float>(dot
                                                  / std::sqrt(referenceEnergy * candidateEnergy))
                             : 0.0f;

    double squaredError = 0.0;
    auto peak = 0.0f;
    auto count = 0;
    for (int channel = 0; channel < channels; ++channel)
    {
        const auto* ref = reference.getReadPointer(channel);
        const auto* model = candidate.getReadPointer(channel);
        for (int sample = start; sample < end; ++sample)
        {
            const auto error = ref[sample]
                               - model[sample + result.lagSamples] * result.fittedGain;
            squaredError += static_cast<double>(error) * error;
            peak = juce::jmax(peak, std::abs(error));
            ++count;
        }
    }
    result.rmsError = count > 0 ? static_cast<float>(std::sqrt(squaredError / count)) : 0.0f;
    result.peakError = peak;
    return result;
}

ReferenceComparator::ExactResult ReferenceComparator::compareExact(
    const juce::AudioBuffer<float>& reference, const juce::AudioBuffer<float>& candidate,
    float tolerance, int maximumLagSamples) noexcept
{
    ExactResult result;
    result.lagSamples = compare(reference, candidate, maximumLagSamples).lagSamples;
    const auto channels = juce::jmin(reference.getNumChannels(), candidate.getNumChannels());
    const auto samples = juce::jmin(reference.getNumSamples(), candidate.getNumSamples());
    const auto start = juce::jmax(0, -result.lagSamples);
    const auto end = juce::jmin(samples, samples - result.lagSamples);
    for (int channel = 0; channel < channels; ++channel)
    {
        const auto* ref = reference.getReadPointer(channel);
        const auto* model = candidate.getReadPointer(channel);
        for (int sample = start; sample < end; ++sample)
        {
            const auto diff = std::abs(ref[sample] - model[sample + result.lagSamples]);
            ++result.compared;
            result.maxAbsDiff = juce::jmax(result.maxAbsDiff, diff);
            if (diff > tolerance)
            {
                ++result.mismatches;
                if (result.firstMismatch < 0 || sample < result.firstMismatch)
                    result.firstMismatch = sample;
            }
        }
    }
    return result;
}
} // namespace wave::dsp
