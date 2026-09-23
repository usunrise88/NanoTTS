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

  /** Which family of graphs this bundle holds, and therefore which backend
   *  reads it. "pocket" is Kyutai Pocket TTS and its fine-tunes; "s3" is the
   *  Supertonic 3 / TeraTTS v2 shape -- text encoder, duration predictor,
   *  flow-matching sampler, vocoder, with voices as style vectors. */
  std::string architecture = "pocket";
  std::string bundle_name;

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

  // --- s3 only ---------------------------------------------------------
  // The sampler works on `s3_latent_dim` channels and the vocoder turns each
  // latent frame into `samples_per_frame` samples. `s3_style_ttl` and
  // `s3_style_dp` are the shapes a voice file must carry, and
  // `s3_vocoder_context` is how many frames of overlap the vocoder needs to
  // decode a chunk without a seam.
  int s3_latent_dim = 144;
  int s3_vocoder_context = 20;
  int s3_stream_chunk = 16;
  float s3_speed = 1.05f;
  // TeraTTS v2 was trained with literal <ru>...</ru> spans in the text and
  // rejects input without them; Supertonic 3 has no language embedding.
  bool s3_language_tags = false;
  std::string s3_default_language = "ru";
  std::vector<int64_t> s3_style_ttl{1, 50, 256};
  std::vector<int64_t> s3_style_dp{1, 8, 16};
  float default_guidance = 3.0f;

  std::vector<StateSpec> flow_state;
  std::vector<StateSpec> mimi_state;

  // bos_before_voice, needed only when warming a voice from audio.
  std::vector<float> bos_before_voice;

  static Bundle load(const std::filesystem::path& dir);
  std::filesystem::path graph(const std::string& name, bool int8) const;
};

}  // namespace nanotts
