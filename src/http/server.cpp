#include "http/server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "audio/analyse.hpp"
#include "audio/mp3.hpp"
#include "audio/resample.hpp"
#include "audio/wav.hpp"
#include "http/web_assets.hpp"
#include "httplib.h"

namespace nanotts {
namespace fs = std::filesystem;
using nlohmann::json;
using clock_t_ = std::chrono::steady_clock;

constexpr double Histogram::kBounds[];

void Histogram::observe(double ms) {
  size_t i = 0;
  while (i < kN && ms > kBounds[i]) ++i;
  buckets_[i].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  double prev = sum_.load(std::memory_order_relaxed);
  while (!sum_.compare_exchange_weak(prev, prev + ms, std::memory_order_relaxed)) {
  }
}

double Histogram::quantile(double q) const {
  const uint64_t total = count_.load(std::memory_order_relaxed);
  if (total == 0) return 0.0;
  const uint64_t target = static_cast<uint64_t>(std::ceil(q * static_cast<double>(total)));
  uint64_t seen = 0;
  for (size_t i = 0; i <= kN; ++i) {
    seen += buckets_[i].load(std::memory_order_relaxed);
    if (seen >= target) return i < kN ? kBounds[i] : kBounds[kN - 1] * 2;
  }
  return kBounds[kN - 1] * 2;
}

std::string Histogram::prometheus(const std::string& name, const std::string& labels) const {
  std::ostringstream os;
  uint64_t cum = 0;
  const std::string sep = labels.empty() ? "" : labels + ",";
  for (size_t i = 0; i < kN; ++i) {
    cum += buckets_[i].load(std::memory_order_relaxed);
    os << name << "_bucket{" << sep << "le=\"" << kBounds[i] << "\"} " << cum << "\n";
  }
  cum += buckets_[kN].load(std::memory_order_relaxed);
  os << name << "_bucket{" << sep << "le=\"+Inf\"} " << cum << "\n";
  os << name << "_sum{" << labels << "} " << sum() << "\n";
  os << name << "_count{" << labels << "} " << count() << "\n";
  return os.str();
}

// ---------------------------------------------------------------- pool

class Service::Lease {
 public:
  Lease(Service* svc, Worker* w) : svc_(svc), w_(w) {}
  ~Lease() {
    if (!w_) return;
    w_->busy.store(false);
    {
      std::lock_guard<std::mutex> lock(svc_->pool_mu_);
      svc_->free_.push_back(w_);
    }
    svc_->pool_cv_.notify_one();
  }
  Lease(Lease&& o) noexcept : svc_(o.svc_), w_(o.w_) { o.w_ = nullptr; }
  Lease(const Lease&) = delete;
  Worker* operator->() const { return w_; }
  Worker& operator*() const { return *w_; }

 private:
  Service* svc_;
  Worker* w_;
};

Service::Lease Service::acquire() {
  queued_.fetch_add(1);
  std::unique_lock<std::mutex> lock(pool_mu_);
  pool_cv_.wait(lock, [&] { return !free_.empty(); });
  Worker* w = free_.back();
  free_.pop_back();
  lock.unlock();
  queued_.fetch_sub(1);
  w->busy.store(true);
  w->served.fetch_add(1);
  return Lease(this, w);
}

const Voice& Service::voice_for(Worker& w, const std::string& id) {
  auto it = w.voices.find(id);
  if (it != w.voices.end()) return *it->second;
  const auto path = store_.path_for(id);
  if (!fs::exists(path)) throw std::runtime_error("unknown voice: " + id);
  auto state = w.engine->load_voice_file(path);
  return *w.voices.emplace(id, std::move(state)).first->second;
}

// ---------------------------------------------------------------- service

struct Service::Impl {
  httplib::Server server;
};

Service::Service(ServerConfig cfg, Bundle bundle)
    : cfg_(std::move(cfg)),
      bundle_(std::move(bundle)),
      topo_(Topology::detect()),
      store_(cfg_.voices_dir),
      impl_(std::make_unique<Impl>()) {
  if (!cfg_.accent_url.empty()) {
    accent_ = std::make_shared<Accentuator>(cfg_.accent_url);
    // Probed rather than required: the sidecar loads models for ten seconds or
    // more, and the synth must not refuse to start because of it.
    std::cout << "accent: " << cfg_.accent_url << (accent_->probe() ? " (ready)" : " (not ready yet)")
              << "\n";
  } else {
    std::cout << "accent: disabled - callers must supply stress themselves\n";
  }

  std::vector<int> nodes = cfg_.nodes;
  if (nodes.empty())
    for (const auto& n : topo_.nodes) nodes.push_back(n.id);

  for (int node : nodes) {
    const NumaNode* info = nullptr;
    for (const auto& n : topo_.nodes)
      if (n.id == node) info = &n;
    if (!info) throw std::runtime_error("no such NUMA node: " + std::to_string(node));

    // Split the node's CPUs between the workers that live on it, so two workers
    // on one node do not fight over the same cores.
    const std::vector<int>& pool = cfg_.use_smt ? info->cpus : info->physical;
    const int per = std::max<int>(1, static_cast<int>(pool.size()) / cfg_.workers_per_node);

    for (int k = 0; k < cfg_.workers_per_node; ++k) {
      EngineConfig ec;
      ec.tokenizer_path = cfg_.tokenizer_path;
      ec.int8 = cfg_.int8;
      ec.numa_node = cfg_.pin ? node : -1;
      if (cfg_.pin) {
        const size_t begin = static_cast<size_t>(k * per);
        for (size_t i = begin; i < pool.size() && i < begin + static_cast<size_t>(per); ++i)
          ec.cpus.push_back(pool[i]);
      }
      ec.threads = cfg_.threads > 0
                       ? cfg_.threads
                       : std::max<int>(1, static_cast<int>(ec.cpus.empty() ? 4 : ec.cpus.size()));
      // Several workers on one machine must not all spin, or they steal each
      // other's cores while idle.
      ec.allow_spinning = nodes.size() * static_cast<size_t>(cfg_.workers_per_node) == 1;
      ec.accent = accent_;

      auto w = std::make_unique<Worker>();
      w->node = node;
      w->engine = make_backend(bundle_, ec);
      std::cout << "worker " << workers_.size() << ": node " << node << ", threads " << ec.threads
                << ", cpus [";
      for (size_t i = 0; i < ec.cpus.size(); ++i) std::cout << (i ? "," : "") << ec.cpus[i];
      std::cout << "]\n";
      free_.push_back(w.get());
      workers_.push_back(std::move(w));
    }
  }
  if (workers_.empty()) throw std::runtime_error("no workers configured");
}

Service::~Service() = default;

namespace {

struct SpeechRequest {
  std::string input;
  std::string voice;
  std::string format = "wav";
  bool stream = true;
  GenParams params;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

const char* content_type_for(const std::string& fmt) {
  if (fmt == "pcm") return "audio/pcm;rate=24000;encoding=float;bits=32";
  if (fmt == "mp3") return "audio/mpeg";
  return "audio/wav";
}

json report_json(const AudioReport& r, int source_rate) {
  json warnings = json::array();
  for (const auto& w : r.warnings) warnings.push_back(w);
  return json{{"source_sample_rate", source_rate},
              {"seconds", r.seconds},
              {"peak", r.peak},
              {"rms", r.rms},
              {"speech_rms", r.speech_rms},
              {"speech_dbfs", r.speech_rms > 0 ? 20.0 * std::log10(r.speech_rms) : -120.0},
              {"dc_offset", r.dc_offset},
              {"clipped_ratio", r.clipped_ratio},
              {"silent_ratio", r.silent_ratio},
              {"crest_db", r.crest_db},
              {"usable", r.usable()},
              {"warnings", warnings}};
}

json error_body(const std::string& msg, const std::string& type, const std::string& code) {
  return {{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
}

}  // namespace

int Service::run() {
  auto& srv = impl_->server;
  srv.set_payload_max_length(cfg_.max_body_bytes);
  srv.set_keep_alive_max_count(100);
  srv.set_read_timeout(30, 0);
  srv.set_write_timeout(120, 0);

  // CORS. A browser page served from another origin (the hosted test console,
  // for instance) cannot talk to this API without it, and a streaming response
  // additionally needs the preflight to pass before any audio flows.
  srv.set_post_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
    const auto origin = req.get_header_value("Origin");
    if (cfg_.cors_origin == "*") {
      res.set_header("Access-Control-Allow-Origin", origin.empty() ? "*" : origin);
      res.set_header("Vary", "Origin");
    } else if (!cfg_.cors_origin.empty()) {
      res.set_header("Access-Control-Allow-Origin", cfg_.cors_origin);
    }
    res.set_header("Access-Control-Allow-Credentials", "true");
  });

  srv.Options(".*", [&](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    const auto ask = req.get_header_value("Access-Control-Request-Headers");
    res.set_header("Access-Control-Allow-Headers", ask.empty() ? "Content-Type, Authorization" : ask);
    res.set_header("Access-Control-Max-Age", "86400");
    // Chrome's Private Network Access check: an HTTPS page reaching a private
    // address is refused unless the preflight opts in.
    if (!req.get_header_value("Access-Control-Request-Private-Network").empty())
      res.set_header("Access-Control-Allow-Private-Network", "true");
    res.status = 204;
  });

  auto authorised = [&](const httplib::Request& req) {
    if (cfg_.api_key.empty()) return true;
    const auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return false;
    const std::string expect = "Bearer " + cfg_.api_key;
    return it->second == expect;
  };

  auto deny = [](httplib::Response& res) {
    res.status = 401;
    res.set_content(error_body("missing or invalid API key", "invalid_request_error",
                               "invalid_api_key")
                        .dump(),
                    "application/json");
  };

  srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string msg = "internal error";
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {
    }
    res.status = 500;
    res.set_content(error_body(msg, "server_error", "internal_error").dump(), "application/json");
  });

  // ---- OpenAI-compatible speech -----------------------------------------
  srv.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    requests_.fetch_add(1);

    SpeechRequest sr;
    sr.params.temperature = bundle_.default_temperature;
    sr.params.eos_threshold = bundle_.default_eos_threshold;
    sr.params.lsd_steps = bundle_.default_lsd_steps;
    try {
      const auto body = json::parse(req.body);
      sr.input = body.at("input").get<std::string>();
      sr.voice = body.value("voice", std::string("male_deep"));
      sr.format = lower(body.value("response_format", std::string("wav")));
      sr.stream = body.value("stream", true);
      // Extra knobs live in their own object so an OpenAI client that does not
      // know about them keeps working.
      // `xvibe` is what this object was called before the project was renamed;
      // it is still accepted so callers written against the old name keep working.
      if (body.contains("nanotts") || body.contains("xvibe")) {
        const auto& x = body.contains("nanotts") ? body.at("nanotts") : body.at("xvibe");
        sr.params.temperature = x.value("temperature", sr.params.temperature);
        sr.params.eos_threshold = x.value("eos_threshold", sr.params.eos_threshold);
        sr.params.lsd_steps = x.value("lsd_steps", sr.params.lsd_steps);
        sr.params.max_frames = x.value("max_frames", sr.params.max_frames);
        sr.params.first_chunk_frames = x.value("first_chunk_frames", sr.params.first_chunk_frames);
        sr.params.chunk_frames = x.value("chunk_frames", sr.params.chunk_frames);
        sr.params.seed = x.value("seed", sr.params.seed);
        sr.params.auto_accent = x.value("auto_accent", sr.params.auto_accent);
      }
    } catch (const std::exception& e) {
      errors_.fetch_add(1);
      res.status = 400;
      res.set_content(error_body(e.what(), "invalid_request_error", "bad_request").dump(),
                      "application/json");
      return;
    }
    if (sr.input.empty()) {
      errors_.fetch_add(1);
      res.status = 400;
      res.set_content(error_body("input is empty", "invalid_request_error", "bad_request").dump(),
                      "application/json");
      return;
    }
    if (sr.format == "mp3" && !Mp3Encoder::available()) {
      errors_.fetch_add(1);
      res.status = 400;
      res.set_content(
          error_body("this build has no mp3 support", "invalid_request_error", "bad_format").dump(),
          "application/json");
      return;
    }

    const auto t_start = clock_t_::now();

    // The provider runs on the connection's thread; it holds the worker for the
    // life of the response, which is what bounds concurrency.
    auto produce = [this, sr, t_start](size_t /*offset*/, httplib::DataSink& sink) {
      auto lease = acquire();
      std::unique_ptr<Mp3Encoder> mp3;
      if (sr.format == "mp3") mp3 = std::make_unique<Mp3Encoder>(bundle_.sample_rate, cfg_.mp3_bitrate);

      bool first = true;
      size_t samples = 0;
      bool wrote_header = false;

      try {
        std::lock_guard<std::mutex> lock(lease->mu);
        const auto& voice = voice_for(*lease, sr.voice);
        lease->engine->generate(sr.input, voice, sr.params, [&](const float* s, size_t n) {
          if (first) {
            ttfb_ms_.observe(
                std::chrono::duration<double, std::milli>(clock_t_::now() - t_start).count());
            first = false;
          }
          samples += n;
          if (sr.format == "pcm") {
            return sink.write(reinterpret_cast<const char*>(s), n * sizeof(float));
          }
          if (sr.format == "mp3") {
            const auto enc = mp3->encode(s, n);
            return enc.empty() ? true
                               : sink.write(reinterpret_cast<const char*>(enc.data()), enc.size());
          }
          if (!wrote_header) {
            // Streaming WAV: length is unknown up front, so the header carries
            // the 0xFFFFFFFF placeholder that players accept.
            const auto h = wav_header(bundle_.sample_rate, 0);
            if (!sink.write(h.data(), h.size())) return false;
            wrote_header = true;
          }
          const auto pcm = wav_encode_pcm16(std::vector<float>(s, s + n), bundle_.sample_rate);
          return sink.write(reinterpret_cast<const char*>(pcm.data()), pcm.size());
        });
        if (mp3) {
          const auto tail = mp3->flush();
          if (!tail.empty()) sink.write(reinterpret_cast<const char*>(tail.data()), tail.size());
        }
      } catch (const std::exception& e) {
        errors_.fetch_add(1);
        std::cerr << "speech failed: " << e.what() << "\n";
        sink.done();
        return false;
      }

      const double total_ms =
          std::chrono::duration<double, std::milli>(clock_t_::now() - t_start).count();
      total_ms_.observe(total_ms);
      const double seconds = static_cast<double>(samples) / bundle_.sample_rate;
      for (auto* acc : {&audio_seconds_}) {
        double prev = acc->load();
        while (!acc->compare_exchange_weak(prev, prev + seconds)) {
        }
      }
      {
        double prev = compute_seconds_.load();
        while (!compute_seconds_.compare_exchange_weak(prev, prev + total_ms / 1000.0)) {
        }
      }
      sink.done();
      return true;
    };

    res.set_chunked_content_provider(content_type_for(sr.format), produce);
  });

  // ---- catalogues --------------------------------------------------------
  srv.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    json data = json::array();
    data.push_back({{"id", bundle_.dir.filename().string()},
                    {"object", "model"},
                    {"owned_by", "nanotts"}});
    res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
  });

  srv.Get("/v1/voices", [&](const httplib::Request& req, httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    json data = json::array();
    for (const auto& v : store_.list()) {
      json entry = {{"id", v.id},
                    {"prefix_frames", v.prefix_frames},
                    {"seconds", static_cast<double>(v.prefix_frames) / bundle_.frame_rate},
                    {"bytes", v.bytes},
                    {"created", v.created}};
      const auto stored = store_.reference_report(v.id);
      entry["reference"] = stored.empty() ? json(nullptr) : json::parse(stored, nullptr, false);
      data.push_back(std::move(entry));
    }
    res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
  });

  // ---- voice warm-up -----------------------------------------------------
  srv.Post("/v1/voices", [&](const httplib::Request& req, httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    if (!req.has_file("file") || !req.has_file("id")) {
      res.status = 400;
      res.set_content(
          error_body("expected multipart fields 'id' and 'file'", "invalid_request_error",
                     "bad_request")
              .dump(),
          "application/json");
      return;
    }
    const std::string id = req.get_file_value("id").content;
    if (!VoiceStore::valid_id(id)) {
      res.status = 400;
      res.set_content(error_body("voice id must be [A-Za-z0-9_-]{1,64}", "invalid_request_error",
                                 "bad_request")
                          .dump(),
                      "application/json");
      return;
    }
    // Architectures that ship fixed style vectors have no encoder to make one
    // from a recording. That is a missing capability, not a bad request, and
    // saying so before reading the upload saves the caller the transfer.
    if (!workers_.empty() && !workers_.front()->engine->can_clone()) {
      res.status = 501;
      res.set_content(error_body("this model ships fixed voices and has no style encoder, so it "
                                 "cannot learn one from a recording; pick one of its own voices",
                                 "invalid_request_error", "cloning_unsupported")
                          .dump(),
                      "application/json");
      return;
    }
    const auto& file = req.get_file_value("file");

    auto lease = acquire();
    std::lock_guard<std::mutex> lock(lease->mu);
    try {
      auto audio = wav_read_memory(reinterpret_cast<const uint8_t*>(file.content.data()),
                                   file.content.size());
      const int src_rate = audio.sample_rate;
      if (audio.sample_rate != bundle_.sample_rate)
        audio.samples = resample(audio.samples, audio.sample_rate, bundle_.sample_rate);

      // Measure the reference before it disappears into the encoder. A warm
      // always succeeds — the model encodes noise just as willingly as speech —
      // so this report is the only thing that tells a bad clone from a good one.
      const auto report = analyse(audio.samples, bundle_.sample_rate);

      const auto t0 = clock_t_::now();
      auto voice = lease->engine->warm_voice(audio.samples);
      const double ms = std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
      lease->engine->save_voice_file(store_.path_for(id), *voice);
      store_.save_reference(id, file.content, report_json(report, src_rate).dump());
      // Invalidate every worker's cached copy. This worker's mutex is already
      // held by the lock above, and std::mutex is not recursive, so it has to
      // be skipped here rather than re-locked.
      for (auto& w : workers_) {
        if (w.get() == &*lease) {
          w->voices.erase(id);
          continue;
        }
        std::lock_guard<std::mutex> wl(w->mu);
        w->voices.erase(id);
      }
      res.set_content(json{{"id", id},
                           {"prefix_frames", voice->prefix_frames},
                           {"seconds", static_cast<double>(voice->prefix_frames) / bundle_.frame_rate},
                           {"source_sample_rate", src_rate},
                           {"warm_ms", ms},
                           {"reference", report_json(report, src_rate)}}
                          .dump(),
                      "application/json");
    } catch (const std::exception& e) {
      errors_.fetch_add(1);
      res.status = 400;
      res.set_content(error_body(e.what(), "invalid_request_error", "warm_failed").dump(),
                      "application/json");
    }
  });

  srv.Get(R"(/v1/voices/([A-Za-z0-9_\-]+)/state)", [&](const httplib::Request& req,
                                                       httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    const std::string id = req.matches[1];
    const auto path = store_.path_for(id);
    if (!fs::exists(path)) {
      res.status = 404;
      res.set_content(error_body("unknown voice", "invalid_request_error", "not_found").dump(),
                      "application/json");
      return;
    }
    std::ifstream f(path, std::ios::binary);
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    res.set_header("Content-Disposition", "attachment; filename=\"" + id + ".safetensors\"");
    res.set_content(body, "application/octet-stream");
  });

  srv.Delete(R"(/v1/voices/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req,
                                                    httplib::Response& res) {
    if (!authorised(req)) return deny(res);
    const std::string id = req.matches[1];
    store_.remove(id);
    for (auto& w : workers_) {
      std::lock_guard<std::mutex> wl(w->mu);
      w->voices.erase(id);
    }
    res.set_content(json{{"deleted", id}}.dump(), "application/json");
  });

  // ---- operations --------------------------------------------------------
  srv.Get("/healthz", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}}.dump(), "application/json");
  });

  srv.Get("/readyz", [&](const httplib::Request&, httplib::Response& res) {
    const bool ready = !workers_.empty();
    res.status = ready ? 200 : 503;
    res.set_content(json{{"ready", ready}, {"workers", workers_.size()}}.dump(),
                    "application/json");
  });

  srv.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
    json w = json::array();
    for (size_t i = 0; i < workers_.size(); ++i)
      w.push_back({{"index", i},
                   {"node", workers_[i]->node},
                   {"busy", workers_[i]->busy.load()},
                   {"served", workers_[i]->served.load()}});
    const double audio = audio_seconds_.load();
    const double compute = compute_seconds_.load();
    res.set_content(json{{"requests", requests_.load()},
                         {"errors", errors_.load()},
                         {"queued", queued_.load()},
                         {"workers", w},
                         {"ttfb_ms", {{"p50", ttfb_ms_.quantile(0.5)},
                                      {"p95", ttfb_ms_.quantile(0.95)},
                                      {"p99", ttfb_ms_.quantile(0.99)}}},
                         {"total_ms", {{"p50", total_ms_.quantile(0.5)},
                                       {"p95", total_ms_.quantile(0.95)},
                                       {"p99", total_ms_.quantile(0.99)}}},
                         {"accent", accent_ ? json{{"url", accent_->url()},
                                                  {"available", accent_->available()},
                                                  {"hits", accent_->stats().hits},
                                                  {"misses", accent_->stats().misses},
                                                  {"failures", accent_->stats().failures},
                                                  {"last_ms", accent_->stats().last_ms}}
                                          : json(nullptr)},
                         {"audio_seconds", audio},
                         {"rtf", compute > 0 && audio > 0 ? compute / audio : 0.0}}
                        .dump(),
                    "application/json");
  });

  srv.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    std::ostringstream os;
    os << "# TYPE nanotts_requests_total counter\nnanotts_requests_total " << requests_.load() << "\n"
       << "# TYPE nanotts_errors_total counter\nnanotts_errors_total " << errors_.load() << "\n"
       << "# TYPE nanotts_queued gauge\nnanotts_queued " << queued_.load() << "\n"
       << "# TYPE nanotts_audio_seconds_total counter\nnanotts_audio_seconds_total "
       << audio_seconds_.load() << "\n"
       << "# TYPE nanotts_ttfb_ms histogram\n" << ttfb_ms_.prometheus("nanotts_ttfb_ms", "")
       << "# TYPE nanotts_request_ms histogram\n" << total_ms_.prometheus("nanotts_request_ms", "");
    for (size_t i = 0; i < workers_.size(); ++i)
      os << "nanotts_worker_served_total{worker=\"" << i << "\",node=\"" << workers_[i]->node
         << "\"} " << workers_[i]->served.load() << "\n";
    res.set_content(os.str(), "text/plain; version=0.0.4");
  });

  // --- console ------------------------------------------------------------
  // A directory wins when one is given, so a developer can point --web at a
  // Vite dev build without rebuilding the binary. Otherwise the bundle
  // compiled into the executable is served straight from memory.
  const auto assets = embedded_web();
  if (!cfg_.web_dir.empty() && fs::exists(cfg_.web_dir)) {
    srv.set_mount_point("/", cfg_.web_dir.string());
    std::cout << "console: " << cfg_.web_dir << " (on disk)\n";
  } else if (!assets.empty()) {
    size_t total = 0;
    for (const auto& a : assets) total += a.body.size();
    std::cout << "console: embedded, " << assets.size() << " files, " << total / 1024
              << " KiB\n";

    srv.Get(".*", [assets](const httplib::Request& req, httplib::Response& res) {
      std::string path = req.path.size() > 1 ? req.path.substr(1) : std::string();

      // Anything that looks like the API must 404 as the API, not as the app.
      if (path.rfind("v1/", 0) == 0 || path == "metrics" || path == "stats") {
        res.status = 404;
        res.set_content(error_body("not found", "invalid_request_error", "not_found").dump(),
                        "application/json");
        return;
      }

      for (const auto& asset : assets) {
        if (asset.path != path) continue;
        // Vite fingerprints asset filenames, so they are safe to cache hard;
        // the entry document must never be.
        res.set_header("Cache-Control", path == "index.html" ? "no-store"
                                                             : "public, max-age=31536000, immutable");
        res.set_content(asset.body.data(), asset.body.size(), std::string(asset.mime).c_str());
        return;
      }

      // Client-side routes (/voices, /monitor) have no file of their own, so
      // the shell answers for them.
      for (const auto& asset : assets) {
        if (asset.path != "index.html") continue;
        res.set_header("Cache-Control", "no-store");
        res.set_content(asset.body.data(), asset.body.size(), std::string(asset.mime).c_str());
        return;
      }
      res.status = 404;
    });
  }

  std::cout.setf(std::ios::unitbuf);  // container logs are piped, not a tty
  std::cout << "listening on http://" << cfg_.host << ":" << cfg_.port << "  (" << workers_.size()
            << " workers)\n";
  if (!srv.listen(cfg_.host, cfg_.port)) {
    std::cerr << "failed to bind " << cfg_.host << ":" << cfg_.port << "\n";
    return 1;
  }
  return 0;
}

}  // namespace nanotts
