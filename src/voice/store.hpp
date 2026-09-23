#pragma once
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace xvibe {

struct VoiceInfo {
  std::string id;
  std::filesystem::path path;
  int64_t prefix_frames = 0;
  uint64_t bytes = 0;
  std::string created;  // ISO-8601, UTC
};

// Flat directory of warmed voices, one .safetensors per voice. Deliberately not
// a database: a voice is a file the operator can copy between machines, and the
// web UI hands the same file back on download.
class VoiceStore {
 public:
  explicit VoiceStore(std::filesystem::path dir);

  std::vector<VoiceInfo> list() const;
  std::optional<VoiceInfo> find(const std::string& id) const;
  std::filesystem::path path_for(const std::string& id) const;
  void remove(const std::string& id);

  // Keeping the upload is what makes a bad clone diagnosable after the fact;
  // without it the only evidence is a state tensor nobody can listen to.
  void save_reference(const std::string& id, const std::string& wav, const std::string& report);
  std::string reference_report(const std::string& id) const;
  std::filesystem::path reference_path(const std::string& id) const;

  // Reject anything that could escape the directory or collide with the shell.
  static bool valid_id(const std::string& id);

 private:
  std::filesystem::path dir_;
  mutable std::mutex mu_;
};

}  // namespace xvibe
