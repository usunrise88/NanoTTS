#pragma once
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "model/bundle.hpp"
#include "text/accent.hpp"
#include "model/state.hpp"
#include "model/tokenizer.hpp"
#include "numa/affinity.hpp"
#include "onnxruntime_cxx_api.h"

namespace nanotts {

struct EngineConfig {
  std::filesystem::path tokenizer_path;
  bool int8 = false;
  int threads = 4;
  int numa_node = -1;          // -1 leaves allocation and pinning alone
  std::vector<int> cpus;       // logical CPUs this engine may run on
  bool allow_spinning = true;  // off when several engines share a machine
  int max_slots = 8;           // batch width of the AR graph
  // Shared across workers so one cache serves the whole service.
  std::shared_ptr<Accentuator> accent;
};

struct GenParams {
  float temperature = 0.5f;
  float eos_threshold = -1.0f;
  int lsd_steps = 1;
  int max_frames = 500;
  int first_chunk_frames = 2;  // small first chunk keeps TTFB down
  int chunk_frames = 15;
  uint64_t seed = 0;  // 0 means "pick one"
  bool auto_accent = true;  // ignored when no sidecar is configured
};

// A warmed voice: the FlowLM cache after the reference audio has been run
// through, plus how many positions it occupies.
struct VoiceState {
  std::unique_ptr<StateBuffers> state;
  int64_t prefix = 0;
};

// Where the wall clock went, in milliseconds. Kept always-on: the counters are
// two integer adds per stage and TTFB work is impossible to direct without them.
struct Timings {
  // `accent` is what TTFB pays; `accent_hidden` is what overlapped with audio.
  double accent = 0, accent_hidden = 0, voice_copy = 0, text_cond = 0, prefill = 0, ar_main = 0, flow = 0,
         decode = 0;
  int frames = 0, chunks = 0;
  void reset() { *this = Timings{}; }
  std::string describe() const;
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

class Engine {
 public:
  Engine(Bundle bundle, EngineConfig cfg);
  ~Engine();

  const Bundle& bundle() const { return bundle_; }
  const Timings& timings() const { return timings_; }
  void reset_timings() { timings_.reset(); }
  const Tokenizer& tokenizer() const { return *tokenizer_; }
  int numa_node() const { return cfg_.numa_node; }

  // Reference audio (mono, bundle sample rate) -> warmed voice state.
  VoiceState warm_voice(const std::vector<float>& audio);

  VoiceState load_voice_file(const std::filesystem::path& p);
  void save_voice_file(const std::filesystem::path& p, const VoiceState& v);
  // Import a genvoice/xVibePocketTTS voice, which uses the upstream layout.
  VoiceState import_upstream_voice(const std::filesystem::path& p);

  // Single-stream synthesis. `on_audio` receives 24 kHz mono float frames as
  // they are decoded and returns false to abort.
  void generate(const std::string& text, const VoiceState& voice, const GenParams& params,
                const std::function<bool(const float*, size_t)>& on_audio);

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
