#include "voice/store.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "voice/safetensors.hpp"

namespace xvibe {
namespace fs = std::filesystem;

namespace {
std::string iso_time(fs::file_time_type t) {
  const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      t - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
  const std::time_t tt = std::chrono::system_clock::to_time_t(sys);
  char buf[32];
  std::tm tm{};
  gmtime_r(&tt, &tm);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}
}  // namespace

bool VoiceStore::valid_id(const std::string& id) {
  if (id.empty() || id.size() > 64) return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '-' || c == '_';
  });
}

VoiceStore::VoiceStore(fs::path dir) : dir_(std::move(dir)) { fs::create_directories(dir_); }

fs::path VoiceStore::path_for(const std::string& id) const {
  if (!valid_id(id)) throw std::runtime_error("invalid voice id");
  return dir_ / (id + ".safetensors");
}

std::vector<VoiceInfo> VoiceStore::list() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<VoiceInfo> out;
  if (!fs::exists(dir_)) return out;
  for (const auto& e : fs::directory_iterator(dir_)) {
    if (!e.is_regular_file() || e.path().extension() != ".safetensors") continue;
    VoiceInfo v;
    v.id = e.path().stem().string();
    v.path = e.path();
    v.bytes = static_cast<uint64_t>(e.file_size());
    v.created = iso_time(e.last_write_time());
    try {
      // prefix_frames lives in the safetensors metadata; the step tensor is the
      // fallback when a file was written by an older build.
      for (const auto& [name, t] : st_load(e.path())) {
        if (name == "step" && t.data.size() >= sizeof(int64_t)) {
          v.prefix_frames = *reinterpret_cast<const int64_t*>(t.data.data());
          break;
        }
      }
    } catch (...) {
      continue;  // not a voice file we understand; skip rather than fail the listing
    }
    out.push_back(std::move(v));
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return out;
}

std::optional<VoiceInfo> VoiceStore::find(const std::string& id) const {
  for (auto& v : list())
    if (v.id == id) return v;
  return std::nullopt;
}

void VoiceStore::remove(const std::string& id) {
  std::lock_guard<std::mutex> lock(mu_);
  fs::remove(path_for(id));
  fs::remove(dir_ / "refs" / (id + ".wav"));
  fs::remove(dir_ / "refs" / (id + ".json"));
}

std::filesystem::path VoiceStore::reference_path(const std::string& id) const {
  if (!valid_id(id)) throw std::runtime_error("invalid voice id");
  return dir_ / "refs" / (id + ".wav");
}

void VoiceStore::save_reference(const std::string& id, const std::string& wav,
                                const std::string& report) {
  std::lock_guard<std::mutex> lock(mu_);
  fs::create_directories(dir_ / "refs");
  std::ofstream audio(reference_path(id), std::ios::binary | std::ios::trunc);
  audio.write(wav.data(), static_cast<std::streamsize>(wav.size()));
  std::ofstream meta(dir_ / "refs" / (id + ".json"), std::ios::trunc);
  meta << report;
}

std::string VoiceStore::reference_report(const std::string& id) const {
  if (!valid_id(id)) return {};
  std::ifstream in(dir_ / "refs" / (id + ".json"));
  if (!in) return {};
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

}  // namespace xvibe
