#include "voice/safetensors.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace nanotts {
namespace fs = std::filesystem;
using nlohmann::json;

static size_t dtype_size(const std::string& d) {
  if (d == "F32" || d == "I32") return 4;
  if (d == "F16" || d == "BF16") return 2;
  if (d == "I64" || d == "U64") return 8;
  if (d == "BOOL" || d == "U8" || d == "I8") return 1;
  throw std::runtime_error("unsupported safetensors dtype: " + d);
}

std::vector<std::pair<std::string, StTensor>> st_load(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());

  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  if (!f || header_len == 0 || header_len > (1ull << 28))
    throw std::runtime_error("bad safetensors header in " + p.string());

  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  json j = json::parse(header);

  const std::streampos data_start = f.tellg();
  std::vector<std::pair<std::string, StTensor>> out;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    StTensor t;
    t.dtype = it.value().at("dtype").get<std::string>();
    for (const auto& d : it.value().at("shape")) t.shape.push_back(d.get<int64_t>());
    const auto offsets = it.value().at("data_offsets");
    const uint64_t begin = offsets[0].get<uint64_t>();
    const uint64_t end = offsets[1].get<uint64_t>();
    if (end < begin) throw std::runtime_error("bad data_offsets for " + it.key());
    t.data.resize(end - begin);
    f.seekg(data_start + static_cast<std::streamoff>(begin));
    f.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(end - begin));
    if (!f) throw std::runtime_error("truncated tensor " + it.key() + " in " + p.string());
    out.emplace_back(it.key(), std::move(t));
  }
  return out;
}

void st_save(const fs::path& p,
             const std::vector<std::pair<std::string, StTensor>>& tensors,
             const std::vector<std::pair<std::string, std::string>>& metadata) {
  json header = json::object();
  if (!metadata.empty()) {
    json m = json::object();
    for (const auto& [k, v] : metadata) m[k] = v;
    header["__metadata__"] = m;
  }

  uint64_t offset = 0;
  for (const auto& [name, t] : tensors) {
    size_t expect = dtype_size(t.dtype);
    for (int64_t d : t.shape) expect *= static_cast<size_t>(d);
    if (expect != t.data.size())
      throw std::runtime_error("tensor " + name + " size does not match its shape");
    header[name] = {{"dtype", t.dtype},
                    {"shape", t.shape},
                    {"data_offsets", {offset, offset + t.data.size()}}};
    offset += t.data.size();
  }

  std::string body = header.dump();
  // The spec wants the payload 8-byte aligned; pad the header with spaces.
  while ((8 + body.size()) % 8 != 0) body.push_back(' ');

  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("cannot write " + p.string());
  const uint64_t hl = body.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(body.data(), static_cast<std::streamsize>(body.size()));
  for (const auto& [name, t] : tensors)
    f.write(reinterpret_cast<const char*>(t.data.data()),
            static_cast<std::streamsize>(t.data.size()));
  if (!f) throw std::runtime_error("failed writing " + p.string());
}

}  // namespace nanotts
