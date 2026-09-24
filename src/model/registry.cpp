#include "model/registry.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace nanotts {
namespace fs = std::filesystem;
namespace {

using nlohmann::json;

double now_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

uint64_t dir_bytes(const fs::path& dir) {
  uint64_t total = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end; it.increment(ec))
    if (!ec && it->is_regular_file(ec)) total += it->file_size(ec);
  return total;
}

}  // namespace

ModelRegistry::ModelRegistry(fs::path root) : root_(std::move(root)) {
  std::error_code ec;
  fs::create_directories(root_, ec);
}

ModelRegistry::~ModelRegistry() {
  std::vector<std::shared_ptr<Running>> pending;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, r] : running_) {
      r->cancel.store(true);
      pending.push_back(r);
    }
    running_.clear();
  }
  for (auto& r : pending)
    if (r->thread.joinable()) r->thread.join();
}

std::vector<InstalledModel> ModelRegistry::installed() const {
  std::vector<InstalledModel> out;
  std::error_code ec;
  if (!fs::exists(root_, ec)) return out;
  for (const auto& e : fs::directory_iterator(root_, ec)) {
    if (ec || !e.is_directory()) continue;
    const fs::path manifest = e.path() / "bundle.json";
    if (!fs::exists(manifest)) continue;

    InstalledModel m;
    m.id = e.path().filename().string();
    m.dir = e.path();
    try {
      std::ifstream in(manifest);
      json j;
      in >> j;
      m.architecture = j.value("architecture", std::string("pocket"));
      m.title = j.value("bundle_name", m.id);
      m.sample_rate = j.value("sample_rate", 0);
    } catch (const std::exception&) {
      continue;  // a half-written directory is not a model
    }
    m.can_clone = m.architecture == "pocket";
    const fs::path voices = e.path() / "voices";
    if (fs::exists(voices, ec))
      for (const auto& v : fs::directory_iterator(voices, ec))
        if (!ec && v.path().extension() == ".safetensors") ++m.voices;
    m.bytes = dir_bytes(e.path());
    out.push_back(std::move(m));
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return out;
}

std::optional<InstalledModel> ModelRegistry::find(const std::string& id) const {
  for (auto& m : installed())
    if (m.id == id) return m;
  return std::nullopt;
}

std::vector<CatalogueEntry> ModelRegistry::available() const {
  const auto have = installed();
  std::vector<CatalogueEntry> out;
  for (const auto& e : fetch_catalogue()) {
    bool present = false;
    for (const auto& m : have) present = present || m.id == e.id;
    if (!present) out.push_back(e);
  }
  return out;
}

std::string ModelRegistry::start_fetch(const std::string& id) {
  const CatalogueEntry* entry = find_in_catalogue(id);
  if (entry == nullptr)
    throw std::runtime_error(
        "unknown model \"" + id +
        "\"; this build can fetch only checkpoints released as ONNX. A Pocket TTS model has to be "
        "traced from its weights, which needs the Python toolchain in install.sh");
  if (find(entry->id)) throw std::runtime_error(entry->id + " is already installed");

  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& [_, j] : jobs_)
    if (j.model == entry->id && j.state == "running")
      throw std::runtime_error(entry->id + " is already being downloaded");

  const std::string job_id = "job_" + std::to_string(next_job_++);
  FetchJob job;
  job.id = job_id;
  job.model = entry->id;
  job.started = now_seconds();
  job.progress.stage = "starting";
  jobs_[job_id] = job;

  auto run = std::make_shared<Running>();
  const CatalogueEntry copy = *entry;
  const fs::path dest = root_ / copy.id;
  // Downloaded into a sibling and moved into place, so a directory under
  // `root_` is either a whole model or absent. A half-finished one would be
  // listed as installed and then fail to load.
  const fs::path staging = root_ / ("." + copy.id + ".partial");

  run->thread = std::thread([this, copy, dest, staging, job_id, run]() {
    std::string error;
    try {
      std::error_code ec;
      fs::remove_all(staging, ec);
      fetch_model(
          copy, staging,
          [&](const FetchProgress& p) {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = jobs_.find(job_id);
            if (it != jobs_.end()) it->second.progress = p;
          },
          run->cancel);
      fs::remove_all(dest, ec);
      fs::rename(staging, dest);
    } catch (const std::exception& e) {
      error = e.what();
      std::error_code ec;
      fs::remove_all(staging, ec);
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job_id);
    if (it != jobs_.end()) {
      it->second.finished = now_seconds();
      if (error.empty()) {
        it->second.state = "done";
      } else if (error == "cancelled" || run->cancel.load()) {
        it->second.state = "cancelled";
      } else {
        it->second.state = "failed";
        it->second.error = error;
      }
    }
    // Deliberately not erased here. A thread that removes its own entry is one
    // the destructor cannot find to join, and the lambda holds `this`: the
    // registry would be free while the thread was still inside it. Finished
    // entries stay joinable until shutdown, which costs a handle each.
  });
  running_[job_id] = run;
  return job_id;
}

std::vector<FetchJob> ModelRegistry::jobs() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<FetchJob> out;
  out.reserve(jobs_.size());
  for (const auto& [_, j] : jobs_) out.push_back(j);
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.started > b.started; });
  return out;
}

std::optional<FetchJob> ModelRegistry::job(const std::string& id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return std::nullopt;
  return it->second;
}

void ModelRegistry::cancel(const std::string& job_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = running_.find(job_id);
  if (it == running_.end()) throw std::runtime_error("no such running job: " + job_id);
  it->second->cancel.store(true);
}

void ModelRegistry::remove(const std::string& id, const std::string& active_id) {
  if (id == active_id)
    throw std::runtime_error("cannot remove " + id + " while it is the model being served");
  const auto m = find(id);
  if (!m) throw std::runtime_error("no such model: " + id);
  std::error_code ec;
  fs::remove_all(m->dir, ec);
  if (ec) throw std::runtime_error("could not remove " + m->dir.string() + ": " + ec.message());
}

}  // namespace nanotts
