#include "audio/wav.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace xvibe {
namespace fs = std::filesystem;

namespace {
uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
uint16_t rd16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}
void wr32(std::string& s, size_t off, uint32_t v) { std::memcpy(s.data() + off, &v, 4); }
void wr16(std::string& s, size_t off, uint16_t v) { std::memcpy(s.data() + off, &v, 2); }
}  // namespace

AudioBuffer wav_read_memory(const uint8_t* data, size_t size) {
  if (size < 44 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0)
    throw std::runtime_error("not a RIFF/WAVE file");

  uint16_t format = 1, channels = 1, bits = 16;
  uint32_t rate = 0;
  const uint8_t* pcm = nullptr;
  uint32_t pcm_bytes = 0;

  size_t off = 12;
  while (off + 8 <= size) {
    const uint32_t id = rd32(data + off);
    uint32_t chunk = rd32(data + off + 4);
    const uint8_t* body = data + off + 8;
    if (body + chunk > data + size) chunk = static_cast<uint32_t>(data + size - body);

    if (id == 0x20746D66u) {  // "fmt "
      if (chunk < 16) throw std::runtime_error("short fmt chunk");
      format = rd16(body);
      channels = rd16(body + 2);
      rate = rd32(body + 4);
      bits = rd16(body + 14);
    } else if (id == 0x61746164u) {  // "data"
      pcm = body;
      pcm_bytes = chunk;
    }
    off += 8 + chunk + (chunk & 1);
  }
  if (!pcm || channels == 0) throw std::runtime_error("wav has no data chunk");

  const size_t bytes_per = bits / 8u;
  const size_t frames = pcm_bytes / (bytes_per * channels);
  AudioBuffer out;
  out.sample_rate = static_cast<int>(rate);
  out.samples.resize(frames);

  for (size_t i = 0; i < frames; ++i) {
    double acc = 0.0;
    for (size_t c = 0; c < channels; ++c) {
      const uint8_t* s = pcm + (i * channels + c) * bytes_per;
      if (format == 3 && bits == 32) {
        float f;
        std::memcpy(&f, s, 4);
        acc += f;
      } else if (bits == 16) {
        int16_t v;
        std::memcpy(&v, s, 2);
        acc += v / 32768.0;
      } else if (bits == 32) {
        int32_t v;
        std::memcpy(&v, s, 4);
        acc += v / 2147483648.0;
      } else if (bits == 24) {
        const int32_t v = (static_cast<int32_t>(static_cast<int8_t>(s[2])) << 16) |
                          (static_cast<int32_t>(s[1]) << 8) | s[0];
        acc += v / 8388608.0;
      } else if (bits == 8) {
        acc += (static_cast<int>(*s) - 128) / 128.0;
      } else {
        throw std::runtime_error("unsupported wav bit depth");
      }
    }
    out.samples[i] = static_cast<float>(acc / channels);
  }
  return out;
}

AudioBuffer wav_read(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return wav_read_memory(buf.data(), buf.size());
}

std::string wav_header(int sample_rate, uint32_t data_bytes) {
  std::string h(44, '\0');
  std::memcpy(h.data(), "RIFF", 4);
  wr32(h, 4, data_bytes ? 36 + data_bytes : 0xFFFFFFFFu);
  std::memcpy(h.data() + 8, "WAVEfmt ", 8);
  wr32(h, 16, 16);
  wr16(h, 20, 1);  // PCM
  wr16(h, 22, 1);  // mono
  wr32(h, 24, static_cast<uint32_t>(sample_rate));
  wr32(h, 28, static_cast<uint32_t>(sample_rate) * 2);
  wr16(h, 32, 2);
  wr16(h, 34, 16);
  std::memcpy(h.data() + 36, "data", 4);
  wr32(h, 40, data_bytes ? data_bytes : 0xFFFFFFFFu);
  return h;
}

std::vector<uint8_t> wav_encode_pcm16(const std::vector<float>& samples, int) {
  std::vector<uint8_t> out(samples.size() * 2);
  for (size_t i = 0; i < samples.size(); ++i) {
    float v = samples[i];
    v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
    const int16_t s = static_cast<int16_t>(v * 32767.0f);
    std::memcpy(out.data() + i * 2, &s, 2);
  }
  return out;
}

void wav_write(const fs::path& p, const std::vector<float>& samples, int sample_rate) {
  const auto pcm = wav_encode_pcm16(samples, sample_rate);
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + p.string());
  const auto h = wav_header(sample_rate, static_cast<uint32_t>(pcm.size()));
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  f.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
}

}  // namespace xvibe
