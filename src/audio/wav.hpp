#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace xvibe {

struct AudioBuffer {
  std::vector<float> samples;  // mono
  int sample_rate = 0;
};

// Minimal RIFF reader: PCM 8/16/24/32-bit and IEEE float, any channel count
// (downmixed by averaging, which is what the upstream loader does).
AudioBuffer wav_read(const std::filesystem::path& p);
AudioBuffer wav_read_memory(const uint8_t* data, size_t size);

std::string wav_header(int sample_rate, uint32_t data_bytes);
std::vector<uint8_t> wav_encode_pcm16(const std::vector<float>& samples, int sample_rate);
void wav_write(const std::filesystem::path& p, const std::vector<float>& samples, int sample_rate);

}  // namespace xvibe
