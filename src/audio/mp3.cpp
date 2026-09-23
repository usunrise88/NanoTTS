#include "audio/mp3.hpp"

#include <stdexcept>

#if NANOTTS_HAVE_LAME
#include <lame/lame.h>
#endif

namespace nanotts {

struct Mp3Encoder::Impl {
#if NANOTTS_HAVE_LAME
  lame_global_flags* gf = nullptr;
  ~Impl() {
    if (gf) lame_close(gf);
  }
#endif
};

bool Mp3Encoder::available() {
#if NANOTTS_HAVE_LAME
  return true;
#else
  return false;
#endif
}

Mp3Encoder::Mp3Encoder(int sample_rate, int bitrate_kbps) : impl_(std::make_unique<Impl>()) {
#if NANOTTS_HAVE_LAME
  impl_->gf = lame_init();
  if (!impl_->gf) throw std::runtime_error("lame_init failed");
  lame_set_in_samplerate(impl_->gf, sample_rate);
  lame_set_out_samplerate(impl_->gf, sample_rate);
  lame_set_num_channels(impl_->gf, 1);
  lame_set_mode(impl_->gf, MONO);
  lame_set_brate(impl_->gf, bitrate_kbps);
  lame_set_quality(impl_->gf, 5);
  lame_set_bWriteVbrTag(impl_->gf, 0);  // no Xing header: the stream has no known length
  if (lame_init_params(impl_->gf) < 0) throw std::runtime_error("lame_init_params failed");
#else
  (void)sample_rate;
  (void)bitrate_kbps;
  throw std::runtime_error("this build has no mp3 support");
#endif
}

Mp3Encoder::~Mp3Encoder() = default;

std::vector<uint8_t> Mp3Encoder::encode(const float* samples, size_t n) {
#if NANOTTS_HAVE_LAME
  std::vector<uint8_t> out(n + n / 4 + 7200);
  // lame_encode_buffer_ieee_float takes normalised [-1,1] samples, unlike
  // lame_encode_buffer_float which wants full-scale +/-32768. Feeding it
  // full-scale values clips everything to silence.
  const int written = lame_encode_buffer_ieee_float(
      impl_->gf, samples, samples, static_cast<int>(n), out.data(),
      static_cast<int>(out.size()));
  if (written < 0) throw std::runtime_error("lame encode failed");
  out.resize(static_cast<size_t>(written));
  return out;
#else
  (void)samples;
  (void)n;
  return {};
#endif
}

std::vector<uint8_t> Mp3Encoder::flush() {
#if NANOTTS_HAVE_LAME
  std::vector<uint8_t> out(7200);
  const int written = lame_encode_flush(impl_->gf, out.data(), static_cast<int>(out.size()));
  out.resize(written > 0 ? static_cast<size_t>(written) : 0);
  return out;
#else
  return {};
#endif
}

}  // namespace nanotts
