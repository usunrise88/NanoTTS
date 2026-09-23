#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nanotts {

struct StTensor {
  std::string dtype;            // F32 | F16 | I64 | BOOL, as safetensors spells it
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

// Minimal safetensors: u64 header length, JSON header, then the payload.
std::vector<std::pair<std::string, StTensor>> st_load(const std::filesystem::path& p);
void st_save(const std::filesystem::path& p,
             const std::vector<std::pair<std::string, StTensor>>& tensors,
             const std::vector<std::pair<std::string, std::string>>& metadata = {});

}  // namespace nanotts
