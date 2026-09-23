#include "audio/analyse.hpp"

#include <algorithm>
#include <cmath>

namespace xvibe {
namespace {

std::string fixed(double v, int digits) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
  return buf;
}

}  // namespace

AudioReport analyse(const std::vector<float>& samples, int sample_rate) {
  AudioReport r;
  if (samples.empty() || sample_rate <= 0) {
    r.warnings.push_back("reference is empty");
    return r;
  }

  r.seconds = static_cast<double>(samples.size()) / sample_rate;

  double sum = 0, sq = 0;
  size_t clipped = 0;
  for (float s : samples) {
    const double v = s;
    sum += v;
    sq += v * v;
    r.peak = std::max(r.peak, std::abs(v));
    if (std::abs(v) >= 0.999) ++clipped;
  }
  r.dc_offset = sum / static_cast<double>(samples.size());
  r.rms = std::sqrt(sq / static_cast<double>(samples.size()));
  r.clipped_ratio = static_cast<double>(clipped) / static_cast<double>(samples.size());

  // Per-frame energy, so silence and speech can be told apart. A single RMS
  // over the whole file cannot: a quiet recording and a half-empty one look
  // identical to it.
  const size_t frame = std::max<size_t>(1, static_cast<size_t>(sample_rate * 0.02));
  std::vector<double> energies;
  energies.reserve(samples.size() / frame + 1);
  for (size_t i = 0; i + frame <= samples.size(); i += frame) {
    double e = 0;
    for (size_t k = 0; k < frame; ++k) e += static_cast<double>(samples[i + k]) * samples[i + k];
    energies.push_back(std::sqrt(e / static_cast<double>(frame)));
  }
  if (energies.empty()) energies.push_back(r.rms);

  const double silence_floor = 0.00316;  // -50 dBFS
  r.silent_ratio =
      static_cast<double>(std::count_if(energies.begin(), energies.end(),
                                        [&](double e) { return e < silence_floor; })) /
      static_cast<double>(energies.size());

  std::vector<double> sorted = energies;
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());
  const size_t take = std::max<size_t>(1, sorted.size() / 2);
  double loud = 0;
  for (size_t i = 0; i < take; ++i) loud += sorted[i] * sorted[i];
  r.speech_rms = std::sqrt(loud / static_cast<double>(take));
  r.crest_db = r.speech_rms > 0 ? 20.0 * std::log10(r.peak / r.speech_rms) : 0.0;

  // Thresholds are advisory: they describe references that measurably degrade
  // cloning, not ones the encoder refuses.
  const double speech_seconds = r.seconds * (1.0 - r.silent_ratio);
  if (speech_seconds < 3.0)
    r.warnings.push_back("only " + fixed(speech_seconds, 1) +
                         " s of speech; 5-20 s of continuous speech clones far better");
  if (r.speech_rms < 0.02)
    r.warnings.push_back("speech level is low (" + fixed(20 * std::log10(std::max(r.speech_rms, 1e-9)), 1) +
                         " dBFS); normalise to about -20 dBFS");
  if (r.clipped_ratio > 0.001)
    r.warnings.push_back(fixed(r.clipped_ratio * 100, 2) +
                         "% of samples are clipped; re-export with headroom");
  if (std::abs(r.dc_offset) > 0.01)
    r.warnings.push_back("DC offset " + fixed(r.dc_offset, 4) + "; high-pass the recording");
  if (r.silent_ratio > 0.6)
    r.warnings.push_back(fixed(r.silent_ratio * 100, 0) +
                         "% of the reference is silence; trim it to the spoken part");
  if (r.crest_db > 24.0)
    r.warnings.push_back("crest factor " + fixed(r.crest_db, 1) +
                         " dB suggests isolated peaks or noise rather than steady speech");
  return r;
}

}  // namespace xvibe
