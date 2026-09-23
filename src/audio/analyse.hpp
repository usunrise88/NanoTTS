#pragma once
#include <string>
#include <vector>

namespace xvibe {

/** What the reference audio actually looks like.
 *
 *  A warm always succeeds — the model will happily encode noise — so the only
 *  way a bad reference surfaces is if someone measures it. These figures are
 *  returned to the caller and stored beside the voice.
 */
struct AudioReport {
  double seconds = 0;
  double peak = 0;          // 0..1
  double rms = 0;           // full-signal RMS
  double speech_rms = 0;    // RMS of the loudest 50% of frames
  double dc_offset = 0;
  double clipped_ratio = 0; // share of samples at or beyond full scale
  double silent_ratio = 0;  // share of 20 ms frames below -50 dBFS
  double crest_db = 0;      // peak over speech RMS
  std::vector<std::string> warnings;

  bool usable() const { return warnings.empty(); }
};

AudioReport analyse(const std::vector<float>& samples, int sample_rate);

}  // namespace xvibe
