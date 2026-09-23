#include "engine/s3.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

#include "engine/engine.hpp"  // Graph
#include "voice/safetensors.hpp"

namespace nanotts {
namespace {

using stopwatch = std::chrono::steady_clock;
double ms_since(stopwatch::time_point t) {
  return std::chrono::duration<double, std::milli>(stopwatch::now() - t).count();
}

std::vector<float> f32_of(const StTensor& t) {
  if (t.dtype != "F32") throw std::runtime_error("style tensor must be float32, got " + t.dtype);
  std::vector<float> out(t.data.size() / sizeof(float));
  std::memcpy(out.data(), t.data.data(), t.data.size());
  return out;
}

int64_t product(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

}  // namespace

struct S3Backend::Impl {
  std::unique_ptr<Ort::Env> env;
  Ort::MemoryInfo mem{nullptr};
  Ort::SessionOptions opts{nullptr};
  std::unique_ptr<Graph> text, duration, sampler, vocoder;
  std::unique_ptr<UnicodeIndexer> indexer;
  S3TextConfig text_cfg;
};

S3Backend::S3Backend(Bundle bundle, EngineConfig cfg)
    : bundle_(std::move(bundle)), cfg_(std::move(cfg)), impl_(std::make_unique<Impl>()) {
  // Same ordering rule as the autoregressive backend: pin and bind before any
  // session exists, because ORT's weights land wherever the creating thread's
  // memory policy points and Linux allocates on first touch.
  if (!cfg_.cpus.empty()) pin_thread(cfg_.cpus);
  const bool bound = cfg_.numa_node >= 0 && bind_memory_to_node(cfg_.numa_node);

  impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nanotts");
  impl_->mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::SessionOptions o;
  o.SetIntraOpNumThreads(cfg_.threads);
  o.SetInterOpNumThreads(1);
  o.SetExecutionMode(ORT_SEQUENTIAL);
  o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  const std::string aff = affinity_string(cfg_.cpus, cfg_.threads);
  if (!aff.empty()) o.AddConfigEntry("session.intra_op_thread_affinities", aff.c_str());
  o.AddConfigEntry("session.intra_op.allow_spinning", cfg_.allow_spinning ? "1" : "0");
  o.AddConfigEntry("session.set_denormal_as_zero", "1");
  // The sampler allocates one latent the size of the whole utterance, so the
  // arena would keep the largest request's worth of memory for the process's
  // life. Bounded reuse matters less here than that does.
  o.DisableCpuMemArena();
  impl_->opts = std::move(o);

  impl_->text = std::make_unique<Graph>(*impl_->env, bundle_.graph("text_encoder", false),
                                        impl_->opts, "text_encoder");
  impl_->duration = std::make_unique<Graph>(*impl_->env, bundle_.graph("duration_predictor", false),
                                            impl_->opts, "duration_predictor");
  impl_->sampler = std::make_unique<Graph>(*impl_->env, bundle_.graph("sampler", cfg_.int8),
                                           impl_->opts, "sampler");
  impl_->vocoder = std::make_unique<Graph>(*impl_->env, bundle_.graph("vocoder", false),
                                           impl_->opts, "vocoder");

  impl_->indexer = std::make_unique<UnicodeIndexer>(bundle_.dir / "unicode_indexer.json");

  impl_->text_cfg.language_tags = bundle_.s3_language_tags;
  impl_->text_cfg.default_language = bundle_.s3_default_language;
  impl_->text_cfg.expand_numbers = !impl_->indexer->supports_digits();
  impl_->text_cfg.stress_plus = true;

  if (bound) unbind_memory();
}

S3Backend::~S3Backend() = default;

VoicePtr S3Backend::load_voice_file(const std::filesystem::path& path) {
  auto voice = std::make_shared<StyleVoice>();
  bool have_ttl = false, have_dp = false;
  for (const auto& [name, t] : st_load(path)) {
    if (name == "style_ttl") {
      voice->style_ttl = f32_of(t);
      voice->ttl_shape = t.shape;
      have_ttl = true;
    } else if (name == "style_dp") {
      voice->style_dp = f32_of(t);
      voice->dp_shape = t.shape;
      have_dp = true;
    }
  }
  if (!have_ttl || !have_dp)
    throw std::runtime_error(path.string() + ": expected style_ttl and style_dp");
  if (voice->ttl_shape != bundle_.s3_style_ttl || voice->dp_shape != bundle_.s3_style_dp)
    throw std::runtime_error(path.string() + ": style shapes do not match this bundle");
  // A style vector has no reference prefix; the catalogue reports 0 rather than
  // inventing a duration.
  voice->prefix_frames = 0;
  return voice;
}

void S3Backend::save_voice_file(const std::filesystem::path& path, const Voice& voice) {
  const auto* v = dynamic_cast<const StyleVoice*>(&voice);
  if (v == nullptr) throw std::runtime_error("voice does not belong to this backend");
  auto blob = [](const std::vector<float>& src, const std::vector<int64_t>& shape) {
    StTensor t;
    t.dtype = "F32";
    t.shape = shape;
    t.data.resize(src.size() * sizeof(float));
    std::memcpy(t.data.data(), src.data(), t.data.size());
    return t;
  };
  st_save(path, {{"style_ttl", blob(v->style_ttl, v->ttl_shape)},
                 {"style_dp", blob(v->style_dp, v->dp_shape)}},
          {{"architecture", "s3"}});
}

VoicePtr S3Backend::warm_voice(const std::vector<float>&) {
  throw std::runtime_error(
      "this checkpoint ships fixed style vectors and no style encoder, so it cannot learn a "
      "voice from a recording; use one of the voices it came with");
}

void S3Backend::generate(const std::string& text, const Voice& voice, const GenParams& params,
                         const std::function<bool(const float*, size_t)>& on_audio) {
  const auto* style = dynamic_cast<const StyleVoice*>(&voice);
  if (style == nullptr) throw std::runtime_error("voice does not belong to this backend");

  const auto t_acc = stopwatch::now();
  const auto prepared = prepare_s3_text(text, *impl_->indexer, impl_->text_cfg,
                                        params.auto_accent ? cfg_.accent.get() : nullptr);
  timings_.accent += ms_since(t_acc);
  if (!prepared.skipped.empty())
    std::fprintf(stderr, "s3: dropped characters absent from the vocabulary: %s\n",
                 prepared.skipped.c_str());

  const auto model_ids = impl_->indexer->encode(prepared.model);
  const auto duration_ids = impl_->indexer->encode(prepared.duration);

  auto run_text = [&](Graph& g, const std::vector<int64_t>& ids, const std::vector<float>& st,
                      const std::vector<int64_t>& st_shape, const char* style_name,
                      std::vector<float>& mask_out) {
    const int64_t len = static_cast<int64_t>(ids.size());
    std::array<int64_t, 2> id_shape{1, len};
    std::array<int64_t, 3> mask_shape{1, 1, len};
    mask_out.assign(static_cast<size_t>(len), 1.f);

    std::vector<const char*> names{"text_ids", style_name, "text_mask"};
    std::vector<Ort::Value> vals;
    vals.push_back(Ort::Value::CreateTensor<int64_t>(impl_->mem, const_cast<int64_t*>(ids.data()),
                                                     ids.size(), id_shape.data(), id_shape.size()));
    vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float*>(st.data()),
                                                   st.size(), st_shape.data(), st_shape.size()));
    vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, mask_out.data(), mask_out.size(),
                                                   mask_shape.data(), mask_shape.size()));
    const char* out_name = g.output_names()[0].c_str();
    return g.session().Run(Ort::RunOptions{nullptr}, names.data(), vals.data(), vals.size(),
                           &out_name, 1);
  };

  const auto t_text = stopwatch::now();
  std::vector<float> text_mask;
  auto emb = run_text(*impl_->text, model_ids, style->style_ttl, style->ttl_shape, "style_ttl",
                      text_mask);
  timings_.text_cond += ms_since(t_text);

  const auto t_dur = stopwatch::now();
  std::vector<float> dur_mask;
  auto dur = run_text(*impl_->duration, duration_ids, style->style_dp, style->dp_shape, "style_dp",
                      dur_mask);
  timings_.duration += ms_since(t_dur);

  const float raw_seconds = dur[0].GetTensorData<float>()[0];
  const double seconds =
      static_cast<double>(raw_seconds) * params.duration_scale / bundle_.s3_speed;
  if (!std::isfinite(seconds) || seconds <= 0)
    throw std::runtime_error("duration predictor returned a non-positive duration");

  const int64_t frames = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(seconds * bundle_.sample_rate / bundle_.samples_per_frame)));
  const int64_t ldim = bundle_.s3_latent_dim;

  // One noise field for the whole utterance: this architecture solves it in a
  // single pass rather than a frame at a time.
  std::mt19937_64 rng(params.seed ? params.seed : std::random_device{}());
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<float> latent(static_cast<size_t>(ldim * frames));
  for (float& x : latent) x = gauss(rng);
  std::vector<float> latent_mask(static_cast<size_t>(frames), 1.f);
  std::array<int64_t, 3> latent_shape{1, ldim, frames};
  std::array<int64_t, 3> latent_mask_shape{1, 1, frames};
  std::array<int64_t, 2> text_mask_shape{1, static_cast<int64_t>(model_ids.size())};
  std::array<int64_t, 3> tm_shape{1, 1, static_cast<int64_t>(model_ids.size())};
  std::array<int64_t, 1> guidance_shape{1};
  float guidance = params.guidance;

  const auto t_sampler = stopwatch::now();
  const char* sampler_in[] = {"initial_latent", "text_emb", "style_ttl", "latent_mask", "text_mask",
                              "guidance"};
  std::vector<Ort::Value> sampler_vals;
  sampler_vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, latent.data(), latent.size(),
                                                         latent_shape.data(), latent_shape.size()));
  sampler_vals.push_back(std::move(emb[0]));
  sampler_vals.push_back(Ort::Value::CreateTensor<float>(
      impl_->mem, const_cast<float*>(style->style_ttl.data()), style->style_ttl.size(),
      style->ttl_shape.data(), style->ttl_shape.size()));
  sampler_vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, latent_mask.data(),
                                                         latent_mask.size(),
                                                         latent_mask_shape.data(),
                                                         latent_mask_shape.size()));
  sampler_vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, text_mask.data(),
                                                         text_mask.size(), tm_shape.data(),
                                                         tm_shape.size()));
  sampler_vals.push_back(Ort::Value::CreateTensor<float>(impl_->mem, &guidance, 1,
                                                         guidance_shape.data(),
                                                         guidance_shape.size()));
  const char* sampler_out = impl_->sampler->output_names()[0].c_str();
  auto solved = impl_->sampler->session().Run(Ort::RunOptions{nullptr}, sampler_in,
                                              sampler_vals.data(), sampler_vals.size(),
                                              &sampler_out, 1);
  timings_.sampler += ms_since(t_sampler);
  timings_.frames = static_cast<int>(frames);

  const float* solved_data = solved[0].GetTensorData<float>();
  const int64_t spf = bundle_.samples_per_frame;
  const int64_t context = bundle_.s3_vocoder_context;
  const int64_t chunk = std::max<int64_t>(1, bundle_.s3_stream_chunk);
  const int64_t max_samples = std::min<int64_t>(
      frames * spf, static_cast<int64_t>(std::llround(seconds * bundle_.sample_rate)));

  // Overlap-save: each chunk is decoded with `context` frames of history in
  // front of it, and that history is thrown away. Concatenating the results
  // matches a single decode of the whole latent.
  const char* voc_in[] = {"latent"};
  const char* voc_out = impl_->vocoder->output_names()[0].c_str();
  int64_t emitted = 0;
  for (int64_t start = 0; start < frames && emitted < max_samples; start += chunk) {
    const int64_t end = std::min(start + chunk, frames);
    const int64_t from = std::max<int64_t>(0, start - context);
    const int64_t width = end - from;

    std::vector<float> slice(static_cast<size_t>(ldim * width));
    for (int64_t c = 0; c < ldim; ++c)
      std::memcpy(slice.data() + c * width, solved_data + c * frames + from,
                  static_cast<size_t>(width) * sizeof(float));

    std::array<int64_t, 3> slice_shape{1, ldim, width};
    const auto t_dec = stopwatch::now();
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->mem, slice.data(), slice.size(),
                                                    slice_shape.data(), slice_shape.size());
    auto wav = impl_->vocoder->session().Run(Ort::RunOptions{nullptr}, voc_in, &in, 1, &voc_out, 1);
    timings_.decode += ms_since(t_dec);
    ++timings_.chunks;

    const float* audio = wav[0].GetTensorData<float>();
    const int64_t produced = wav[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const int64_t discard = (start - from) * spf;
    int64_t want = (end - start) * spf;
    if (discard + want > produced) want = produced - discard;
    if (want <= 0) continue;
    want = std::min(want, max_samples - emitted);
    if (!on_audio(audio + discard, static_cast<size_t>(want))) return;
    emitted += want;
  }
}

std::unique_ptr<Backend> make_s3_backend(Bundle bundle, EngineConfig cfg) {
  return std::make_unique<S3Backend>(std::move(bundle), std::move(cfg));
}

}  // namespace nanotts
