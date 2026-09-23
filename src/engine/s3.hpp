#pragma once
#include <memory>
#include <mutex>

#include "engine/backend.hpp"
#include "text/s3_text.hpp"

namespace nanotts {

/** The Supertonic 3 / TeraTTS v2 family.
 *
 *  Four graphs and no recurrence:
 *
 *      text_encoder(text_ids, style_ttl, text_mask)          -> text_emb
 *      duration_predictor(text_ids, style_dp, text_mask)     -> seconds
 *      sampler(initial_latent, text_emb, style_ttl,
 *              latent_mask, text_mask, guidance)             -> latent
 *      vocoder(latent)                                        -> audio
 *
 *  The sampler owns its whole diffusion schedule, so swapping in a graph with
 *  a different step count or distillation changes the model without touching
 *  this code -- which is how both checkpoints ship two samplers.
 *
 *  What this costs, and it is the honest headline: the duration predictor sizes
 *  the entire utterance and the sampler fills all of it in one pass, so nothing
 *  can be heard until that pass is done. TTFB therefore grows with the length
 *  of the text, where the autoregressive backend's does not. Only the vocoder
 *  streams, in overlap-save chunks.
 *
 *  Voices are the style vectors that ship with the checkpoint. There is no
 *  style encoder in the release, so `can_clone()` is false: this family cannot
 *  learn a voice from a recording the way Pocket TTS can.
 */
class S3Backend : public Backend {
 public:
  S3Backend(Bundle bundle, EngineConfig cfg);
  ~S3Backend() override;

  const Bundle& bundle() const override { return bundle_; }
  const Timings& timings() const override { return timings_; }
  void reset_timings() override { timings_.reset(); }
  int numa_node() const override { return cfg_.numa_node; }

  void generate(const std::string& text, const Voice& voice, const GenParams& params,
                const std::function<bool(const float*, size_t)>& on_audio) override;

  VoicePtr load_voice_file(const std::filesystem::path& path) override;
  void save_voice_file(const std::filesystem::path& path, const Voice& voice) override;

  bool can_clone() const override { return false; }
  VoicePtr warm_voice(const std::vector<float>& audio) override;

 private:
  struct Impl;
  Bundle bundle_;
  EngineConfig cfg_;
  Timings timings_;
  std::unique_ptr<Impl> impl_;
};

/** A voice for this family: the two style tensors, exactly as shipped. */
struct StyleVoice : Voice {
  std::vector<float> style_ttl;  // [1, 50, 256] for the released checkpoints
  std::vector<float> style_dp;   // [1, 8, 16]
  std::vector<int64_t> ttl_shape, dp_shape;
};

}  // namespace nanotts
