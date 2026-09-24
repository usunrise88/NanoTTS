#include "engine/engine.hpp"

#include <chrono>
#include <condition_variable>
#include <future>
#include <deque>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "text/text.hpp"
#include "voice/safetensors.hpp"

namespace nanotts {
namespace fs = std::filesystem;

namespace {

// How much text is accented before the first chunk can start, as a percentage
// of the chunk budget. Stress adds a token per stressed word -- measured at
// 1.38x median on Russian prose -- so ~70% of the budget unaccented is about one
// full chunk once stress lands on it.
constexpr int kAccentLeadPercent = 70;

using stopwatch = std::chrono::steady_clock;
double ms_since(stopwatch::time_point t) {
  return std::chrono::duration<double, std::milli>(stopwatch::now() - t).count();
}

// fp16 helpers. Only the voice import path needs conversion; the graphs keep
// the cache in fp16 end to end.
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);          // flush subnormals
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // inf / nan -> inf
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    out = sign;
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}


}  // namespace


// ---------------------------------------------------------------- Graph

Graph::Graph(Ort::Env& env, const fs::path& path, const Ort::SessionOptions& opts,
             const char* label) {
  if (!fs::exists(path)) throw std::runtime_error(std::string(label) + ": missing " + path.string());
  session_ = std::make_unique<Ort::Session>(env, path.c_str(), opts);
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < session_->GetInputCount(); ++i)
    input_names_.emplace_back(session_->GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < session_->GetOutputCount(); ++i)
    output_names_.emplace_back(session_->GetOutputNameAllocated(i, alloc).get());
}

int Graph::input_index(const std::string& name) const {
  for (size_t i = 0; i < input_names_.size(); ++i)
    if (input_names_[i] == name) return static_cast<int>(i);
  return -1;
}

int Graph::output_index(const std::string& name) const {
  for (size_t i = 0; i < output_names_.size(); ++i)
    if (output_names_[i] == name) return static_cast<int>(i);
  return -1;
}

// ---------------------------------------------------------------- Engine

Engine::Engine(Bundle bundle, EngineConfig cfg)
    : bundle_(std::move(bundle)), cfg_(std::move(cfg)) {
  alloc_ = std::make_unique<NumaAllocator>(cfg_.numa_node);
  init_sessions();
  tokenizer_ = std::make_unique<Tokenizer>(cfg_.tokenizer_path);
  scratch_noise_.resize(static_cast<size_t>(bundle_.latent_dim));
  scratch_x_.resize(static_cast<size_t>(bundle_.latent_dim));
}

Engine::~Engine() = default;

void Engine::init_sessions() {
  // Pin and bind *before* the sessions exist: ORT's weights land wherever the
  // creating thread's memory policy points, and Linux allocates on first touch.
  if (!cfg_.cpus.empty()) pin_thread(cfg_.cpus);
  const bool bound = cfg_.numa_node >= 0 && bind_memory_to_node(cfg_.numa_node);

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nanotts");
  mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::string aff = affinity_string(cfg_.cpus, cfg_.threads);

  auto make = [&](bool arena) {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(cfg_.threads);
    o.SetInterOpNumThreads(1);
    o.SetExecutionMode(ORT_SEQUENTIAL);
    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!aff.empty()) o.AddConfigEntry("session.intra_op_thread_affinities", aff.c_str());
    o.AddConfigEntry("session.intra_op.allow_spinning", cfg_.allow_spinning ? "1" : "0");
    o.AddConfigEntry("session.set_denormal_as_zero", "1");
    if (!arena) {
      // Only the encoder opts out: it sees up to 30 s of audio in one call and
      // ORT's arena never hands memory back to the OS, so a single big request
      // would permanently inflate RSS. The decoder's chunks are bounded, and
      // leaving its memory pattern enabled is worth a large share of its time.
      o.DisableCpuMemArena();
      o.DisableMemPattern();
    }
    return o;
  };
  opts_ar_ = make(true);
  opts_aux_ = make(false);

  g_text_ = std::make_unique<Graph>(*env_, bundle_.graph("text_conditioner", false), opts_ar_,
                                    "text_conditioner");
  g_main_ = std::make_unique<Graph>(*env_, bundle_.graph("flow_lm_main", cfg_.int8), opts_ar_,
                                    "flow_lm_main");
  g_flow_ = std::make_unique<Graph>(*env_, bundle_.graph("flow_lm_flow", cfg_.int8), opts_ar_,
                                    "flow_lm_flow");
  g_enc_ = std::make_unique<Graph>(*env_, bundle_.graph("mimi_encoder", false), opts_aux_,
                                   "mimi_encoder");
  g_dec_ = std::make_unique<Graph>(*env_, bundle_.graph("mimi_decoder", false), opts_ar_,
                                   "mimi_decoder");

  if (bound) unbind_memory();
  warmup();
}

void Engine::warmup() {
  // ORT lazily prepacks weights and plans memory on the first run of each
  // graph. Paying for that here keeps it out of the first request's TTFB.
  try {
    StateBuffers flow(bundle_.flow_state, 1, alloc_.get());
    StateBuffers mimi(bundle_.mimi_state, 1, alloc_.get());
    flow.reset();
    mimi.reset();

    std::vector<float> text(static_cast<size_t>(bundle_.cond_dim), 0.f);
    const float empty = 0.f;
    run_main(&empty, 0, text.data(), 1, flow, 1);

    std::vector<float> seq(static_cast<size_t>(bundle_.latent_dim), 0.f);
    auto run = run_main(seq.data(), 1, &empty, 0, flow, 1);
    std::mt19937_64 rng(1);
    run_flow_steps(run.conditioning, 1, 0.f, rng);

    std::vector<float> latent(static_cast<size_t>(bundle_.latent_dim), 0.f);
    std::vector<float> pcm;
    decode_frames(latent, 1, mimi, pcm);
  } catch (const std::exception& e) {
    // A warmup failure is not fatal; the real request will surface the error.
    std::fprintf(stderr, "warmup skipped: %s\n", e.what());
  }
  timings_.reset();
}

Engine::FlowRun Engine::run_main(const float* sequence, int64_t seq_len, const float* text,
                                 int64_t text_len, StateBuffers& state, int64_t batch) {
  const int64_t ldim = bundle_.latent_dim;
  const int64_t cdim = bundle_.cond_dim;

  std::array<int64_t, 3> seq_shape{batch, seq_len, ldim};
  std::array<int64_t, 3> text_shape{batch, text_len, cdim};

  std::vector<const char*> in_names, out_names;
  std::vector<Ort::Value> in_vals, out_vals;
  in_names.reserve(state.count() + 2);
  in_vals.reserve(state.count() + 2);

  in_names.push_back("sequence");
  in_vals.push_back(Ort::Value::CreateTensor<float>(
      mem_, const_cast<float*>(sequence), static_cast<size_t>(batch * seq_len * ldim),
      seq_shape.data(), seq_shape.size()));
  in_names.push_back("text_embeddings");
  in_vals.push_back(Ort::Value::CreateTensor<float>(
      mem_, const_cast<float*>(text), static_cast<size_t>(batch * text_len * cdim),
      text_shape.data(), text_shape.size()));

  for (size_t i = 0; i < state.count(); ++i) {
    in_names.push_back(state.spec(i).name.c_str());
    in_vals.push_back(state.input(i, mem_));
  }

  static const char* kCond = "conditioning";
  static const char* kEos = "eos_logit";
  out_names.push_back(kCond);
  out_names.push_back(kEos);
  out_vals.push_back(Ort::Value{nullptr});
  out_vals.push_back(Ort::Value{nullptr});

  std::vector<std::string> out_state_names;
  out_state_names.reserve(state.count());
  for (size_t i = 0; i < state.count(); ++i) out_state_names.push_back("out_" + state.spec(i).name);
  for (size_t i = 0; i < state.count(); ++i) {
    out_names.push_back(out_state_names[i].c_str());
    out_vals.push_back(state.output(i, mem_));
  }

  g_main_->session().Run(Ort::RunOptions{nullptr}, in_names.data(), in_vals.data(), in_vals.size(),
                         out_names.data(), out_vals.data(), out_vals.size());
  state.swap();

  const size_t cond_n = static_cast<size_t>(batch) * static_cast<size_t>(cdim);
  scratch_cond_.resize(cond_n);
  std::memcpy(scratch_cond_.data(), out_vals[0].GetTensorData<float>(), cond_n * sizeof(float));
  static thread_local std::vector<float> eos_copy;
  eos_copy.assign(out_vals[1].GetTensorData<float>(),
                  out_vals[1].GetTensorData<float>() + static_cast<size_t>(batch));
  return FlowRun{scratch_cond_.data(), eos_copy.data()};
}

std::vector<float> Engine::run_flow_steps(const float* cond, int lsd_steps, float temperature,
                                          std::mt19937_64& rng) {
  const int64_t ldim = bundle_.latent_dim;
  const float stddev = temperature > 0.f ? std::sqrt(temperature) : 0.f;

  std::vector<float> x(static_cast<size_t>(ldim), 0.f);
  if (stddev > 0.f) {
    std::normal_distribution<float> gauss(0.f, stddev);
    for (auto& v : x) v = gauss(rng);
  }

  const float dt = 1.0f / static_cast<float>(lsd_steps);
  std::array<int64_t, 2> c_shape{1, bundle_.cond_dim};
  std::array<int64_t, 2> s_shape{1, 1};
  std::array<int64_t, 2> x_shape{1, ldim};
  const char* in_names[] = {"c", "s", "t", "x"};
  const char* out_names[] = {"flow_dir"};

  for (int j = 0; j < lsd_steps; ++j) {
    float s = static_cast<float>(j) / static_cast<float>(lsd_steps);
    float t = static_cast<float>(j + 1) / static_cast<float>(lsd_steps);
    Ort::Value vals[4] = {
        Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(cond),
                                        static_cast<size_t>(bundle_.cond_dim), c_shape.data(), 2),
        Ort::Value::CreateTensor<float>(mem_, &s, 1, s_shape.data(), 2),
        Ort::Value::CreateTensor<float>(mem_, &t, 1, s_shape.data(), 2),
        Ort::Value::CreateTensor<float>(mem_, x.data(), x.size(), x_shape.data(), 2)};
    auto out = g_flow_->session().Run(Ort::RunOptions{nullptr}, in_names, vals, 4, out_names, 1);
    const float* dir = out[0].GetTensorData<float>();
    for (int64_t i = 0; i < ldim; ++i) x[static_cast<size_t>(i)] += dir[i] * dt;
  }
  return x;
}

void Engine::init_decoder_names(const StateBuffers& mimi) {
  dec_out_storage_.clear();
  dec_out_storage_.reserve(mimi.count());
  for (size_t i = 0; i < mimi.count(); ++i) dec_out_storage_.push_back("out_" + mimi.spec(i).name);

  dec_in_storage_.clear();
  dec_in_storage_.reserve(mimi.count() + 1);
  dec_in_storage_.push_back("latent");
  for (size_t i = 0; i < mimi.count(); ++i) dec_in_storage_.push_back(mimi.spec(i).name);

  dec_in_names_.clear();
  dec_in_names_.reserve(dec_in_storage_.size());
  for (const auto& n : dec_in_storage_) dec_in_names_.push_back(n.c_str());

  dec_out_names_.clear();
  dec_out_names_.reserve(dec_out_storage_.size() + 1);
  dec_out_names_.push_back("audio");
  for (const auto& n : dec_out_storage_) dec_out_names_.push_back(n.c_str());
}

void Engine::decode_frames(const std::vector<float>& latents, int64_t frames, StateBuffers& mimi,
                           std::vector<float>& out) {
  if (dec_in_names_.empty()) init_decoder_names(mimi);

  std::array<int64_t, 3> shape{1, frames, bundle_.latent_dim};
  std::vector<Ort::Value> in_vals;
  in_vals.reserve(mimi.count() + 1);
  in_vals.push_back(Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(latents.data()),
                                                    latents.size(), shape.data(), shape.size()));
  for (size_t i = 0; i < mimi.count(); ++i) in_vals.push_back(mimi.input(i, mem_));

  std::vector<Ort::Value> out_vals;
  out_vals.reserve(mimi.count() + 1);
  out_vals.push_back(Ort::Value{nullptr});
  for (size_t i = 0; i < mimi.count(); ++i) out_vals.push_back(mimi.output(i, mem_));

  g_dec_->session().Run(Ort::RunOptions{nullptr}, dec_in_names_.data(), in_vals.data(),
                        in_vals.size(), dec_out_names_.data(), out_vals.data(), out_vals.size());
  mimi.swap();

  const auto info = out_vals[0].GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  const float* audio = out_vals[0].GetTensorData<float>();
  out.assign(audio, audio + n);
}

VoicePtr Engine::warm_voice(const std::vector<float>& audio) {
  // The encoder graph was traced frame-aligned, so the caller's audio is padded
  // up to a whole frame before it goes in.
  const size_t spf = static_cast<size_t>(bundle_.samples_per_frame);
  std::vector<float> padded = audio;
  if (padded.size() % spf) padded.resize(padded.size() + (spf - padded.size() % spf), 0.f);
  if (padded.empty()) throw std::runtime_error("reference audio is empty");

  std::array<int64_t, 3> shape{1, 1, static_cast<int64_t>(padded.size())};
  const char* in_names[] = {"audio"};
  const char* out_names[] = {"conditioning"};
  Ort::Value in = Ort::Value::CreateTensor<float>(mem_, padded.data(), padded.size(), shape.data(),
                                                  shape.size());
  auto out = g_enc_->session().Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

  const auto info = out[0].GetTensorTypeAndShapeInfo().GetShape();
  const int64_t frames = info[1];
  const int64_t cdim = bundle_.cond_dim;
  const float* cond = out[0].GetTensorData<float>();

  std::vector<float> prompt;
  int64_t prompt_frames = frames;
  if (bundle_.insert_bos_before_voice && !bundle_.bos_before_voice.empty()) {
    prompt_frames += 1;
    prompt.reserve(static_cast<size_t>(prompt_frames * cdim));
    prompt.insert(prompt.end(), bundle_.bos_before_voice.begin(), bundle_.bos_before_voice.end());
    prompt.insert(prompt.end(), cond, cond + frames * cdim);
  } else {
    prompt.assign(cond, cond + frames * cdim);
  }
  if (prompt_frames > bundle_.max_seq)
    throw std::runtime_error("reference audio is too long for this bundle's max_seq");

  auto v = std::make_shared<VoiceState>();
  v->state = std::make_unique<StateBuffers>(bundle_.flow_state, 1, alloc_.get());
  v->state->reset();
  const float empty_seq = 0.f;
  run_main(&empty_seq, 0, prompt.data(), prompt_frames, *v->state, 1);
  v->prefix = prompt_frames;
  v->prefix_frames = prompt_frames;
  return v;
}

VoicePtr Engine::load_voice_file(const fs::path& p) {
  const auto tensors = st_load(p);
  VoiceState v;
  v.state = std::make_unique<StateBuffers>(bundle_.flow_state, 1, alloc_.get());
  v.state->reset();
  for (size_t i = 0; i < bundle_.flow_state.size(); ++i) {
    const auto& spec = bundle_.flow_state[i];
    bool found = false;
    for (const auto& [name, t] : tensors) {
      if (name != spec.name) continue;
      if (t.data.size() != v.state->slot_bytes(i))
        throw std::runtime_error("voice tensor " + name + " has the wrong size for this bundle");
      std::memcpy(v.state->slot_ptr(i, 0), t.data.data(), t.data.size());
      found = true;
      break;
    }
    if (!found) throw std::runtime_error("voice file is missing tensor " + spec.name);
  }
  const int64_t* step = reinterpret_cast<const int64_t*>(v.state->slot_ptr(0, 0));
  v.prefix = *step;
  v.prefix_frames = v.prefix;
  return std::make_shared<VoiceState>(std::move(v));
}

void Engine::save_voice_file(const fs::path& p, const Voice& voice) {
  const auto* vp = dynamic_cast<const VoiceState*>(&voice);
  if (vp == nullptr) throw std::runtime_error("voice does not belong to this backend");
  const VoiceState& v = *vp;
  std::vector<std::pair<std::string, StTensor>> tensors;
  for (size_t i = 0; i < bundle_.flow_state.size(); ++i) {
    const auto& spec = bundle_.flow_state[i];
    StTensor t;
    t.dtype = spec.dtype == "float16" ? "F16" : (spec.dtype == "int64" ? "I64" : "F32");
    t.shape = spec.shape;
    t.shape[0] = 1;
    t.data.assign(v.state->slot_ptr(i, 0), v.state->slot_ptr(i, 0) + v.state->slot_bytes(i));
    tensors.emplace_back(spec.name, std::move(t));
  }
  st_save(p, tensors,
          {{"bundle", bundle_.dir.filename().string()},
           {"prefix_frames", std::to_string(v.prefix)}});
}

VoicePtr Engine::import_upstream_voice(const fs::path& p) {
  // Upstream stores transformer.layers.N.self_attn/{offset,pad,cache} with the
  // cache as [2,1,T,H,Dh] fp32. Here K and V are separate fp16 buffers of
  // capacity max_seq and the position is one shared vector.
  const auto tensors = st_load(p);
  auto find = [&](const std::string& key) -> const StTensor* {
    for (const auto& [name, t] : tensors)
      if (name == key) return &t;
    return nullptr;
  };

  VoiceState v;
  v.state = std::make_unique<StateBuffers>(bundle_.flow_state, 1, alloc_.get());
  v.state->reset();

  int64_t prefix = -1;
  const int64_t H = bundle_.num_heads, Dh = bundle_.dim_per_head;
  for (int layer = 0; layer < bundle_.num_layers; ++layer) {
    const std::string base = "transformer.layers." + std::to_string(layer) + ".self_attn";
    const StTensor* cache = find(base + "/cache");
    const StTensor* offset = find(base + "/offset");
    if (!cache || !offset) throw std::runtime_error("not an upstream voice file: " + p.string());
    if (cache->dtype != "F32" || cache->shape.size() != 5)
      throw std::runtime_error("unexpected cache layout in " + p.string());

    const int64_t T = cache->shape[2];
    if (T > bundle_.max_seq)
      throw std::runtime_error("voice prefix " + std::to_string(T) + " exceeds max_seq");
    const int64_t off = *reinterpret_cast<const int64_t*>(offset->data.data());
    if (prefix < 0) prefix = off;
    if (off != prefix) throw std::runtime_error("layers disagree on offset in " + p.string());

    const float* src = reinterpret_cast<const float*>(cache->data.data());
    const size_t per_kv = static_cast<size_t>(T * H * Dh);
    for (int kv = 0; kv < 2; ++kv) {
      const size_t idx = 1 + static_cast<size_t>(layer) * 2 + static_cast<size_t>(kv);
      uint16_t* dst = reinterpret_cast<uint16_t*>(v.state->slot_ptr(idx, 0));
      const float* s = src + static_cast<size_t>(kv) * per_kv;
      for (size_t i = 0; i < per_kv; ++i) dst[i] = f32_to_f16(s[i]);
    }
  }
  *reinterpret_cast<int64_t*>(v.state->slot_ptr(0, 0)) = prefix;
  v.prefix = prefix;
  v.prefix_frames = prefix;
  (void)f16_to_f32;
  return std::make_shared<VoiceState>(std::move(v));
}

void Engine::generate(const std::string& text, const Voice& voice_in, const GenParams& params,
                      const std::function<bool(const float*, size_t)>& on_audio) {
  const auto* voice_p = dynamic_cast<const VoiceState*>(&voice_in);
  if (voice_p == nullptr) throw std::runtime_error("voice does not belong to this backend");
  const VoiceState& voice = *voice_p;
  // `+vowel` becomes U+0301 first, so hand-written stress is already in the
  // model's notation before the sidecar sees it — and so the sidecar can tell
  // which words to leave alone.
  // Characters the checkpoint has no token for become <unk> otherwise, which
  // the model reads as noise in the middle of a phrase. Done before chunking,
  // while the text is still whole.
  // Numbers first, while they are still digits, then the vocabulary guard.
  // This checkpoint cannot read digits at all -- they come out as noise and
  // take the neighbouring words with them -- so spelling them is part of
  // making it legible, not a convenience.
  std::string dropped;
  const std::string prepared = map_to_vocabulary(
      spell_numbers(to_model_stress(text), bundle_.language), *tokenizer_, dropped);
  if (!dropped.empty())
    std::fprintf(stderr, "text: dropped characters this checkpoint has no token for: %s\n",
                 dropped.c_str());
  const bool accenting = params.auto_accent && cfg_.accent != nullptr;

  // Only as much text as the first chunk needs is accented on the critical path;
  // the rest is accented on another thread while that chunk is being spoken.
  // The cut is on a sentence boundary and chunking happens *after* accentuation,
  // so the chunk boundaries are the ones the unsplit text would have produced.
  // Chunking first and accenting the chunks would be cheaper but wrong: the
  // budget would have to be guessed down to cover the stress tokens, which
  // breaks long sentences at commas the model would rather have read through.
  auto chunk_it = [this](const std::string& t) {
    return split_chunks(*tokenizer_, t, bundle_.max_token_per_chunk, bundle_.pad_with_spaces,
                        bundle_.remove_semicolons);
  };

  std::vector<std::string> chunks;
  std::future<std::vector<std::string>> tail_chunks;
  if (accenting) {
    auto [head, tail] =
        split_lead(*tokenizer_, prepared, (bundle_.max_token_per_chunk * kAccentLeadPercent) / 100,
                   bundle_.pad_with_spaces, bundle_.remove_semicolons);
    if (!tail.empty()) {
      tail_chunks = std::async(std::launch::async, [this, chunk_it, rest = std::move(tail)]() {
        return chunk_it(cfg_.accent->apply(rest));
      });
    }
    const auto t_acc = stopwatch::now();
    const std::string accented = cfg_.accent->apply(head);
    timings_.accent += ms_since(t_acc);
    chunks = chunk_it(accented);
  } else {
    chunks = chunk_it(prepared);
  }

  std::mt19937_64 rng(params.seed ? params.seed : std::random_device{}());
  StateBuffers work(bundle_.flow_state, 1, alloc_.get());
  StateBuffers mimi(bundle_.mimi_state, 1, alloc_.get());

  const int64_t ldim = bundle_.latent_dim;

  // The AR step is memory-bound and the Mimi decoder is compute-bound, so
  // running them back to back wastes most of a core. The decoder lives on its
  // own thread and the AR loop hands it finished frames; audio still leaves
  // through `on_audio` on the caller's thread, which keeps the HTTP sink
  // single-threaded.
  struct Pipe {
    std::mutex mu;
    std::condition_variable cv_in, cv_out;
    std::deque<std::vector<float>> latents;  // one entry per chunk, already grouped
    std::deque<std::vector<float>> audio;
    bool closed = false, aborted = false;
    std::string error;
  } pipe;

  std::thread decoder([&] {
    std::vector<float> pcm;
    for (;;) {
      std::vector<float> block;
      {
        std::unique_lock<std::mutex> lock(pipe.mu);
        pipe.cv_in.wait(lock, [&] { return !pipe.latents.empty() || pipe.closed || pipe.aborted; });
        if (pipe.aborted) return;
        if (pipe.latents.empty()) {
          if (pipe.closed) return;
          continue;
        }
        block = std::move(pipe.latents.front());
        pipe.latents.pop_front();
      }
      const int64_t frames = static_cast<int64_t>(block.size()) / ldim;
      try {
        const auto t = stopwatch::now();
        decode_frames(block, frames, mimi, pcm);
        timings_.decode += ms_since(t);
        ++timings_.chunks;
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(pipe.mu);
        pipe.aborted = true;
        pipe.error = e.what();
        pipe.cv_out.notify_all();
        return;
      }
      {
        std::lock_guard<std::mutex> lock(pipe.mu);
        pipe.audio.push_back(pcm);
      }
      pipe.cv_out.notify_all();
    }
  });

  // Hand finished audio to the caller. `wait` blocks until the decoder has
  // drained, which is how the tail of a request is flushed.
  auto drain = [&](bool wait) {
    for (;;) {
      std::vector<float> out;
      {
        std::unique_lock<std::mutex> lock(pipe.mu);
        if (wait)
          pipe.cv_out.wait(lock, [&] {
            return !pipe.audio.empty() || pipe.aborted ||
                   (pipe.closed && pipe.latents.empty());
          });
        if (pipe.aborted) return false;
        if (pipe.audio.empty()) return true;
        out = std::move(pipe.audio.front());
        pipe.audio.pop_front();
      }
      if (!on_audio(out.data(), out.size())) {
        std::lock_guard<std::mutex> lock(pipe.mu);
        pipe.aborted = true;
        pipe.cv_in.notify_all();
        return false;
      }
    }
  };

  auto shutdown = [&](bool abort) {
    {
      std::lock_guard<std::mutex> lock(pipe.mu);
      if (abort) pipe.aborted = true;
      pipe.closed = true;
    }
    pipe.cv_in.notify_all();
    pipe.cv_out.notify_all();
    if (decoder.joinable()) decoder.join();
  };

  bool aborted = false;
  try {
    for (size_t ci = 0; !aborted; ++ci) {
      // The tail is joined only once the head has been spoken, which is the
      // latest possible moment and so hides the most of its cost.
      if (ci == chunks.size()) {
        if (!tail_chunks.valid()) break;
        const auto t_join = stopwatch::now();
        const auto more = tail_chunks.get();
        timings_.accent_hidden += ms_since(t_join);
        chunks.insert(chunks.end(), more.begin(), more.end());
        if (ci == chunks.size()) break;
      }
      const std::string& chunk = chunks[ci];
      auto [prepared, guess] =
          prepare_text_prompt(chunk, bundle_.pad_with_spaces, bundle_.remove_semicolons);
      const int frames_after_eos = guess + 2;

      const auto t_tc = stopwatch::now();
      const auto ids = tokenizer_->encode(prepared);
      std::vector<int64_t> ids64(ids.begin(), ids.end());
      std::array<int64_t, 2> tok_shape{1, static_cast<int64_t>(ids64.size())};
      const char* tc_in[] = {"token_ids"};
      const char* tc_out[] = {"embeddings"};
      Ort::Value tok = Ort::Value::CreateTensor<int64_t>(mem_, ids64.data(), ids64.size(),
                                                         tok_shape.data(), tok_shape.size());
      auto emb = g_text_->session().Run(Ort::RunOptions{nullptr}, tc_in, &tok, 1, tc_out, 1);
      const float* text_emb = emb[0].GetTensorData<float>();
      const int64_t text_len = static_cast<int64_t>(ids64.size());
      timings_.text_cond += ms_since(t_tc);

      // Upstream bounds a chunk's generation by its token count -- 3 tokens per
      // second of speech plus two seconds of slack -- and warns when the loop
      // ends without EOS. A flat cap lets a chunk that never emits EOS babble
      // for the full budget, which is what a missed EOS sounds like.
      const int chunk_max_frames = std::min<int>(
          params.max_frames,
          static_cast<int>(std::ceil((static_cast<double>(text_len) / 3.0 + 2.0) * bundle_.frame_rate)));

      const auto t_copy = stopwatch::now();
      work.load_slot(0, *voice.state, 0);
      timings_.voice_copy += ms_since(t_copy);

      const float empty = 0.f;
      const auto t_pre = stopwatch::now();
      run_main(&empty, 0, text_emb, text_len, work, 1);
      timings_.prefill += ms_since(t_pre);

      std::vector<float> cur(static_cast<size_t>(ldim),
                             std::numeric_limits<float>::quiet_NaN());  // NaN == BOS
      std::vector<float> pending;
      int64_t pending_frames = 0;
      int eos_at = -1;
      int want = params.first_chunk_frames;

      for (int step = 0; step < chunk_max_frames; ++step) {
        const auto t_ar = stopwatch::now();
        auto run = run_main(cur.data(), 1, &empty, 0, work, 1);
        timings_.ar_main += ms_since(t_ar);
        if (eos_at < 0 && run.eos_logit[0] > params.eos_threshold) eos_at = step;
        if (eos_at >= 0 && step >= eos_at + frames_after_eos) break;

        const auto t_flow = stopwatch::now();
        cur = run_flow_steps(run.conditioning, params.lsd_steps, params.temperature, rng);
        timings_.flow += ms_since(t_flow);
        ++timings_.frames;

        pending.insert(pending.end(), cur.begin(), cur.end());
        ++pending_frames;

        if (pending_frames >= want) {
          {
            std::lock_guard<std::mutex> lock(pipe.mu);
            if (pipe.aborted) { aborted = true; break; }
            pipe.latents.push_back(std::move(pending));
          }
          pipe.cv_in.notify_one();
          pending.clear();
          pending_frames = 0;
          want = params.chunk_frames;
        }
        if (!drain(false)) { aborted = true; break; }
      }

      if (!aborted && pending_frames > 0) {
        {
          std::lock_guard<std::mutex> lock(pipe.mu);
          pipe.latents.push_back(std::move(pending));
        }
        pipe.cv_in.notify_one();
      }
      // Each text chunk restarts the Mimi state, so the decoder must be idle
      // before the next one begins.
      if (!aborted) {
        std::unique_lock<std::mutex> lock(pipe.mu);
        pipe.cv_out.wait(lock, [&] { return pipe.latents.empty() || pipe.aborted; });
      }
      if (!aborted && !drain(false)) aborted = true;
    }
  } catch (...) {
    shutdown(true);
    throw;
  }

  if (!aborted) {
    // Close first, then drain: the decoder exits once the queue is empty.
    {
      std::lock_guard<std::mutex> lock(pipe.mu);
      pipe.closed = true;
    }
    pipe.cv_in.notify_all();
    if (decoder.joinable()) decoder.join();
    drain(false);
  } else {
    shutdown(true);
  }

  std::string err;
  {
    std::lock_guard<std::mutex> lock(pipe.mu);
    err = pipe.error;
  }
  if (!err.empty()) throw std::runtime_error("mimi decode failed: " + err);
}

}  // namespace nanotts
