#pragma once
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/engine.hpp"
#include "numa/topology.hpp"
#include "text/accent.hpp"
#include "voice/store.hpp"

namespace xvibe {

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8080;
  std::filesystem::path bundle_dir = "bundle";
  std::filesystem::path tokenizer_path;
  std::filesystem::path voices_dir = "voices";
  std::filesystem::path web_dir = "src/web";
  bool int8 = false;

  // Worker layout. `nodes` empty means "one worker per detected NUMA node".
  std::vector<int> nodes;
  int workers_per_node = 1;
  int threads = 0;        // 0 -> physical cores available to the worker
  bool use_smt = false;   // give workers their SMT siblings too
  bool pin = true;
  std::string api_key;    // empty disables auth
  // Origin allowed to call the API from a browser. "*" opens it to any page,
  // which is what a hosted test console needs; set it explicitly in production.
  std::string cors_origin = "*";
  // Empty leaves stress to the caller; the quality cost is large and measured.
  std::string accent_url;
  int mp3_bitrate = 96;
  size_t max_body_bytes = 32ull << 20;
};

// Latency histogram with fixed millisecond buckets. Good enough for the p50/p95
// the service needs to expose, and cheap enough to update on every request.
class Histogram {
 public:
  void observe(double ms);
  double quantile(double q) const;
  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  double sum() const { return sum_.load(std::memory_order_relaxed); }
  std::string prometheus(const std::string& name, const std::string& labels) const;

 private:
  static constexpr double kBounds[] = {5,   10,  25,   50,   75,   100,  150,  200,
                                       300, 500, 750,  1000, 2000, 5000, 10000};
  static constexpr size_t kN = sizeof(kBounds) / sizeof(kBounds[0]);
  mutable std::atomic<uint64_t> buckets_[kN + 1]{};
  std::atomic<uint64_t> count_{0};
  std::atomic<double> sum_{0.0};
};

// One engine plus everything that must not be shared across NUMA nodes: its
// voice states are allocated by that engine's node-local allocator, so a warmed
// voice is cached per worker rather than globally.
struct Worker {
  std::unique_ptr<Engine> engine;
  std::mutex mu;
  std::unordered_map<std::string, VoiceState> voices;
  int node = -1;
  std::atomic<bool> busy{false};
  std::atomic<uint64_t> served{0};
};

class Service {
 public:
  Service(ServerConfig cfg, Bundle bundle);
  ~Service();

  int run();

 private:
  class Lease;
  Lease acquire();

  const VoiceState& voice_for(Worker& w, const std::string& id);

  ServerConfig cfg_;
  Bundle bundle_;
  Topology topo_;
  VoiceStore store_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::shared_ptr<Accentuator> accent_;

  std::mutex pool_mu_;
  std::condition_variable pool_cv_;
  std::vector<Worker*> free_;

  Histogram ttfb_ms_, total_ms_;
  std::atomic<uint64_t> requests_{0}, errors_{0}, queued_{0};
  std::atomic<double> audio_seconds_{0.0}, compute_seconds_{0.0};

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xvibe
