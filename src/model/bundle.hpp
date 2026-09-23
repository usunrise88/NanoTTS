#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nanotts {

// One recurrent tensor as the export declared it. `shape[0]` is the batch axis
// and comes through as -1; the runtime substitutes the slot count.
struct StateSpec {
  std::string name;   // graph input name ("cache_k_0", "state_7", ...)
  std::string path;   // upstream module path, kept for mimi so dumps stay readable
  std::string dtype;  // float16 | float32 | int64 | bool
  std::vector<int64_t> shape;

  int64_t elements(int64_t batch) const;
  size_t element_size() const;
};

struct Bundle {
  std::filesystem::path dir;

  int sample_rate = 24000;
  int samples_per_frame = 1920;
  double frame_rate = 12.5;
  int latent_dim = 32;
  int cond_dim = 1024;
  int num_layers = 6;
  int num_heads = 16;
  int dim_per_head = 64;
  int max_seq = 512;
  int mimi_cache_len = 250;
  int vocab_size = 5000;
  int max_token_per_chunk = 50;
  bool insert_bos_before_voice = true;
  bool pad_with_spaces = false;
  bool remove_semicolons = false;

  float default_temperature = 0.5f;
  float default_eos_threshold = -1.0f;
  int default_lsd_steps = 1;

  std::vector<StateSpec> flow_state;
  std::vector<StateSpec> mimi_state;

  // bos_before_voice, needed only when warming a voice from audio.
  std::vector<float> bos_before_voice;

  static Bundle load(const std::filesystem::path& dir);
  std::filesystem::path graph(const std::string& name, bool int8) const;
};

}  // namespace nanotts
