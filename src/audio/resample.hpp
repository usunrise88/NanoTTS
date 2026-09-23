#pragma once
#include <vector>

namespace xvibe {
// Polyphase FIR with a Kaiser window, matching scipy.signal.resample_poly's
// defaults closely enough that voice cloning from non-24 kHz references does
// not drift. Reference audio is the only place this runs.
std::vector<float> resample(const std::vector<float>& in, int from_rate, int to_rate);
}  // namespace xvibe
