#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace nanotts {

// Streaming MP3 encoder. Present because OpenAI TTS clients default to mp3;
// pcm remains the low-latency path.
class Mp3Encoder {
 public:
  Mp3Encoder(int sample_rate, int bitrate_kbps);
  ~Mp3Encoder();
  Mp3Encoder(const Mp3Encoder&) = delete;
  Mp3Encoder& operator=(const Mp3Encoder&) = delete;

  std::vector<uint8_t> encode(const float* samples, size_t n);
  std::vector<uint8_t> flush();
  static bool available();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nanotts
