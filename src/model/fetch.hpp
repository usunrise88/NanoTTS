#pragma once
#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace nanotts {

/** Downloads a checkpoint that ships ONNX and turns it into a bundle.
 *
 *  Only the s3 family can be installed this way, and that is not an omission:
 *  Pocket TTS is released as weights, so preparing one means tracing five
 *  graphs through PyTorch. That needs a Python toolchain the serving image
 *  deliberately does not carry, so those models stay with the installer and
 *  this path reports why rather than pretending.
 */

struct FetchProgress {
  std::string stage;      // what is happening, for the console to show
  std::string file;       // which file, when one is being transferred
  uint64_t done = 0;      // bytes of the current file
  uint64_t total = 0;     // its size, 0 when the server did not say
  int step = 0, steps = 0;  // position in the whole job
};

using ProgressFn = std::function<void(const FetchProgress&)>;

/** One entry of the catalogue of models that can be installed on demand. */
struct CatalogueEntry {
  std::string id;            // what the API calls it: "TeraTTSv2"
  std::string repo;          // "TeraSpace/TeraTTSv2"
  std::string architecture;  // "s3"
  std::string title;
  std::string languages;
  bool can_clone = false;
  uint64_t approx_bytes = 0;
};

/** The models this build knows how to fetch by itself. */
const std::vector<CatalogueEntry>& fetch_catalogue();
const CatalogueEntry* find_in_catalogue(const std::string& id_or_repo);

/** Downloads `entry` into `dir`, writing the bundle manifest and the voices.
 *  `cancel` is polled between files so a job can be stopped. Throws with a
 *  message meant to be shown to whoever asked for the install. */
void fetch_model(const CatalogueEntry& entry, const std::filesystem::path& dir,
                 const ProgressFn& progress, const std::atomic<bool>& cancel);

}  // namespace nanotts
