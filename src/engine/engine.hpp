#pragma once
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "engine/backend.hpp"
#include "model/bundle.hpp"
#include "text/accent.hpp"
#include "model/state.hpp"
#include "model/tokenizer.hpp"
#include "numa/affinity.hpp"
#include "onnxruntime_cxx_api.h"

namespace nanotts {

// A warmed voice for this backend: the FlowLM cache after the reference audio
// has been run through, plus how many positions it occupies.
struct VoiceState : Voice {
  std::unique_ptr<StateBuffers> state;
  int64_t prefix = 0;
};

// Thin wrapper that resolves graph I/O names to indices once, so the AR loop
// never does string lookups. PocketTTS.cpp calls std::string::find on every
// input name of every step; at 12.5 frames a second across 20 tensors that is
// pure overhead.
class Graph {
 public:
  Graph(Ort::Env& env, const std::filesystem::path& path, const Ort::SessionOptions& opts,
        const char* label);

  Ort::Session& session() { return *session_; }
  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }
  int input_index(const std::string& name) const;
  int output_index(const std::string& name) const;

 private:
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_, output_names_;
};

class Engine : public Backend {
 public:
  Engine(Bundle bundle, EngineConfig cfg);
  ~Engine() override;

  const Bundle& bundle() const override { return bundle_; }
  const Timings& timings() const override { return timings_; }
  void reset_timings() override { timings_.reset(); }
  const Tokenizer& tokenizer() const { return *tokenizer_; }
  int numa_node() const override { return cfg_.numa_node; }
  bool can_clone() const override { return true; }

  // Reference audio (mono, bundle sample rate) -> warmed voice state.
  VoicePtr warm_voice(const std::vector<float>& audio) override;

  VoicePtr load_voice_file(const std::filesystem::path& p) override;
  void save_voice_file(const std::filesystem::path& p, const Voice& v) override;
  // Import a genvoice/xVibePocketTTS voice, which uses the upstream layout.
  VoicePtr import_upstream_voice(const std::filesystem::path& p);

  // Single-stream synthesis. `on_audio` receives mono float frames at the
  // bundle's sample rate as they are decoded and returns false to abort.
  void generate(const std::string& text, const Voice& voice, const GenParams& params,
                const std::function<bool(const float*, size_t)>& on_audio) override;

 private:
  struct FlowRun {
    const float* conditioning;
    const float* eos_logit;
  };

  void init_sessions();
  void warmup();
  // One pass through flow_lm_main. `seq_len`/`text_len` may be zero, which is
  // how prefill and AR steps share a graph.
  FlowRun run_main(const float* sequence, int64_t seq_len, const float* text, int64_t text_len,
                   StateBuffers& state, int64_t batch);
  std::vector<float> run_flow_steps(const float* cond, int lsd_steps, float temperature,
                                    std::mt19937_64& rng);
  void decode_frames(const std::vector<float>& latents, int64_t frames, StateBuffers& mimi,
                     std::vector<float>& out);

  Bundle bundle_;
  EngineConfig cfg_;
  std::unique_ptr<NumaAllocator> alloc_;
  std::unique_ptr<Ort::Env> env_;
  Ort::MemoryInfo mem_{nullptr};
  Ort::SessionOptions opts_ar_, opts_aux_;

  std::unique_ptr<Graph> g_text_, g_main_, g_flow_, g_enc_, g_dec_;
  std::unique_ptr<Tokenizer> tokenizer_;

  Timings timings_;

  // Scratch reused across frames so the hot loop allocates nothing.
  std::vector<float> scratch_noise_, scratch_x_, scratch_cond_;
};

}  // namespace nanotts
