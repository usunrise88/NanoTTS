#pragma once
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "model/fetch.hpp"

namespace nanotts {

/** A model the server can run: one directory holding a bundle and its voices. */
struct InstalledModel {
  std::string id;
  std::filesystem::path dir;
  std::string architecture;
  std::string title;
  int sample_rate = 0;
  int voices = 0;
  uint64_t bytes = 0;
  bool can_clone = false;
};

/** A download in flight. Jobs are kept after they finish so the console can
 *  show what happened without having to have been watching. */
struct FetchJob {
  std::string id;
  std::string model;
  std::string state = "running";  // running | done | failed | cancelled
  std::string error;
  FetchProgress progress;
  double started = 0, finished = 0;
};

/** Tracks what is installed, downloads what is not, and hands the service a
 *  directory to switch to.
 *
 *  Installing is deliberately asynchronous: a checkpoint is hundreds of
 *  megabytes, and an HTTP request that blocks for four minutes is a request
 *  that times out somewhere in the middle and leaves no way to find out how it
 *  went.
 */
class ModelRegistry {
 public:
  explicit ModelRegistry(std::filesystem::path root);
  ~ModelRegistry();

  /** Everything on disk, newest scan. */
  std::vector<InstalledModel> installed() const;
  /** Catalogue entries that are not installed yet. */
  std::vector<CatalogueEntry> available() const;
  std::optional<InstalledModel> find(const std::string& id) const;

  /** Starts a download. Returns the job id, or throws when the model is
   *  unknown, already installed, or of a family this build cannot fetch. */
  std::string start_fetch(const std::string& id);
  std::vector<FetchJob> jobs() const;
  std::optional<FetchJob> job(const std::string& id) const;
  void cancel(const std::string& job_id);

  /** Removes an installed model. Refuses the one currently in use; that is the
   *  service's business, so it passes in which one that is. */
  void remove(const std::string& id, const std::string& active_id);

  const std::filesystem::path& root() const { return root_; }

 private:
  struct Running {
    std::thread thread;
    std::atomic<bool> cancel{false};
  };

  std::filesystem::path root_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, FetchJob> jobs_;
  std::unordered_map<std::string, std::shared_ptr<Running>> running_;
  uint64_t next_job_ = 1;
};

}  // namespace nanotts
