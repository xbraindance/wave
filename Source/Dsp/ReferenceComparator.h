#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

namespace wave::dsp
{
class ReferenceComparator
{
public:
    struct Metrics
    {
        int lagSamples = 0;
        float fittedGain = 1.0f;
        float correlation = 0.0f;
        float rmsError = 0.0f;
        float peakError = 0.0f;
    };

    // Sample-exact mode for ASIC fitting: no fitted gain, aligned at the correlation lag.
    // `tolerance` is in full-scale units (0 = bit-exact; use 0.5 / 512 for half an LSB of a 10-bit bus).
    struct ExactResult
    {
        int lagSamples = 0;
        int compared = 0;
        int mismatches = 0;
        int firstMismatch = -1; // reference sample index, -1 = none
        float maxAbsDiff = 0.0f;
    };

    [[nodiscard]] static ExactResult compareExact(const juce::AudioBuffer<float>& reference,
                                                  const juce::AudioBuffer<float>& candidate,
                                                  float tolerance = 0.0f,
                                                  int maximumLagSamples = 2048) noexcept;

    [[nodiscard]] static Metrics compare(const juce::AudioBuffer<float>& reference,
                                         const juce::AudioBuffer<float>& candidate,
                                         int maximumLagSamples = 2048) noexcept;
};
} // namespace wave::dsp
