#pragma once
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model/bundle.hpp"
#include "numa/affinity.hpp"
#include "text/accent.hpp"

namespace nanotts {

/** What a synthesis backend has to provide.
 *
 *  Two architectures live behind this, and they agree on almost nothing below
 *  the waterline:
 *
 *  - **pocket** (Kyutai Pocket TTS and its fine-tunes) is autoregressive. It
 *    emits one 12.5 Hz latent at a time through a KV cache, so audio starts
 *    flowing after the first frame and TTFB does not grow with the length of
 *    the text. A voice is that cache, prefilled from reference audio, which is
 *    also why it can clone from a WAV.
 *  - **s3** (Supertonic 3, TeraTTS v2) is not. A duration predictor sizes the
 *    whole utterance, one flow-matching pass fills it in, and only the vocoder
 *    streams. TTFB therefore grows with the text, and a voice is a pair of
 *    style vectors that ship with the checkpoint -- there is no encoder to make
 *    one from your own recording.
 *
 *  Everything above the waterline -- the HTTP API, streaming, wav/pcm/mp3, NUMA
 *  placement, metrics, the console -- is the same for both, which is the point
 *  of the split.
 */

/** A warmed voice. The contents belong to the backend that made it. */
struct Voice {
  virtual ~Voice() = default;
  /** Reference length in frames where the architecture has such a notion;
   *  zero when it does not, which the catalogue reports as an unknown length. */
  int64_t prefix_frames = 0;
};
using VoicePtr = std::shared_ptr<const Voice>;

/** Generation knobs. The two groups are per-architecture and a backend ignores
 *  the ones that are not its own; keeping them in one struct means the HTTP
 *  layer does not have to know which model is loaded to parse a request. */
struct GenParams {
  // pocket
  float temperature = 0.5f;
  float eos_threshold = -4.0f;
  int lsd_steps = 1;
  int max_frames = 500;
  // One latent frame per decoder call, which is how upstream drives Mimi and,
  // as it turns out, how it sounds best. Feeding it fifteen at a time is not
  // the same computation -- the decoder's output depends on the grouping -- and
  // measured against ASR the difference is not subtle: WER 4.62% against 6.39%,
  // CER 1.29% against 2.38%, on the same twenty phrases and seeds. It is also
  // the lower latency of the two, so there is nothing to trade off.
  int first_chunk_frames = 1;
  int chunk_frames = 1;
  // s3
  float guidance = 3.0f;        // classifier-free guidance for the sampler
  float duration_scale = 1.0f;  // >1 slows the whole utterance down
  // both
  uint64_t seed = 0;        // 0 means "pick one"
  bool auto_accent = true;  // ignored when no sidecar is configured
};

struct EngineConfig {
  std::filesystem::path tokenizer_path;
  bool int8 = false;
  int threads = 4;
  int numa_node = -1;          // -1 leaves allocation and pinning alone
  std::vector<int> cpus;       // logical CPUs this engine may run on
  bool allow_spinning = true;  // off when several engines share a machine
  int max_slots = 8;           // batch width of the AR graph, where there is one
  // Shared across workers so one cache serves the whole service.
  std::shared_ptr<Accentuator> accent;
};

/** Where the wall clock went, in milliseconds. Kept always-on: the counters are
 *  two integer adds per stage and TTFB work is impossible to direct without
 *  them. Not every stage exists in every architecture; the unused ones stay at
 *  zero and `describe` leaves them out. */
struct Timings {
  // `accent` is what TTFB pays; `accent_hidden` is what overlapped with audio.
  double accent = 0, accent_hidden = 0, voice_copy = 0, text_cond = 0, prefill = 0, ar_main = 0,
         flow = 0, decode = 0;
  // s3 only: the duration predictor and the one-shot sampler pass.
  double duration = 0, sampler = 0;
  int frames = 0, chunks = 0;
  void reset() { *this = Timings{}; }
  std::string describe() const;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual const Bundle& bundle() const = 0;
  virtual const Timings& timings() const = 0;
  virtual void reset_timings() = 0;
  virtual int numa_node() const = 0;

  /** Single-stream synthesis. `on_audio` receives mono float frames at the
   *  bundle's sample rate as they are decoded, and returns false to abort. */
  virtual void generate(const std::string& text, const Voice& voice, const GenParams& params,
                        const std::function<bool(const float*, size_t)>& on_audio) = 0;

  virtual VoicePtr load_voice_file(const std::filesystem::path& path) = 0;
  virtual void save_voice_file(const std::filesystem::path& path, const Voice& voice) = 0;

  /** False when the architecture ships fixed voices and has no encoder to make
   *  one from a recording. The API answers 501 rather than pretending. */
  virtual bool can_clone() const = 0;
  /** Reference audio (mono, bundle sample rate) -> warmed voice.
   *  Throws when `can_clone()` is false. */
  virtual VoicePtr warm_voice(const std::vector<float>& audio) = 0;
};

/** ORT's `session.intra_op_thread_affinities` spelling for a CPU set. Shared:
 *  every backend pins its sessions the same way. */
std::string affinity_string(const std::vector<int>& cpus, int threads);

/** Builds the backend the bundle asks for. Throws on an architecture this
 *  binary does not implement, naming what it found. */
std::unique_ptr<Backend> make_backend(Bundle bundle, EngineConfig cfg);

}  // namespace nanotts
