#include "model/fetch.hpp"

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "httplib.h"
#include "voice/safetensors.hpp"

namespace nanotts {
namespace fs = std::filesystem;
namespace {

using nlohmann::json;

constexpr const char* kHost = "https://huggingface.co";
constexpr int64_t kSamplesPerFrame = 3072;

/** Layout of a release: where its graphs, styles and character table live, and
 *  what its text layer needs. Both known families put the same four graphs in
 *  different folders and spell their voices differently. */
struct Layout {
  const char* graph_dir;
  const char* indexer;
  const char* styles_dir;
  bool styles_are_npy;
  std::vector<const char*> sampler_candidates;
  bool language_tags;
  const char* default_language;
  int sample_rate;
  float speed;
};

const Layout kTera{"models",
                   "unicode_indexer.json",
                   "styles",
                   true,
                   {"sampler_distilled_cfg3_8step.onnx", "sampler_teacher_8step.onnx"},
                   true,
                   "ru",
                   44100,
                   1.05f};
const Layout kSupertonic{"onnx",
                         "onnx/unicode_indexer.json",
                         "voice_styles",
                         false,
                         {"vector_estimator.onnx"},
                         false,
                         "en",
                         44100,
                         1.0f};

/** Splits an absolute URL into the origin a client is built from and the
 *  path-with-query it is asked for. */
void split_url(const std::string& url, std::string& origin, std::string& path) {
  const auto scheme_end = url.find("://");
  if (scheme_end == std::string::npos) throw std::runtime_error("not an absolute URL: " + url);
  const auto host_end = url.find('/', scheme_end + 3);
  if (host_end == std::string::npos) {
    origin = url;
    path = "/";
  } else {
    origin = url.substr(0, host_end);
    path = url.substr(host_end);
  }
}

httplib::Client client_for(const std::string& origin) {
  httplib::Client c(origin);
  // Redirects are followed by hand below. cpp-httplib re-encodes the location
  // it was given, and HuggingFace hands out CDN URLs whose signature covers the
  // exact percent-encoding -- re-encoding it turns a valid link into a 403.
  c.set_follow_location(false);
  // And not re-encoded either: a CDN signature covers the exact spelling of the
  // query it was issued for, so percent-encoding it again invalidates it.
  c.set_url_encode(false);
  c.set_connection_timeout(20, 0);
  c.set_read_timeout(180, 0);
  c.set_write_timeout(60, 0);
  c.set_default_headers({{"User-Agent", "nanotts"}});
  return c;
}

constexpr int kMaxRedirects = 5;

/** GET that follows redirects itself, handing `sink` each chunk of the final
 *  response. Returns the status of that response. */
int http_get(const std::string& url, const httplib::ContentReceiver& sink,
             const httplib::Progress& on_progress) {
  std::string next = url;
  for (int hop = 0; hop <= kMaxRedirects; ++hop) {
    std::string origin, path;
    split_url(next, origin, path);
    auto c = client_for(origin);

    std::string location;
    int status = 0;
    auto r = c.Get(
        path.c_str(), httplib::Headers{},
        [&](const httplib::Response& response) {
          status = response.status;
          if (response.has_header("Location")) location = response.get_header_value("Location");
          // A redirect body is not the file; drop it rather than feeding the sink.
          return true;
        },
        [&](const char* data, size_t len) {
          if (status >= 300 && status < 400) return true;
          return sink(data, len);
        },
        on_progress);
    if (!r)
      throw std::runtime_error("cannot reach " + origin + " (" + httplib::to_string(r.error()) +
                               "); installing a model needs outbound HTTPS");
    if (status >= 300 && status < 400 && !location.empty()) {
      next = location.rfind("http", 0) == 0 ? location : origin + location;
      continue;
    }
    return status;
  }
  throw std::runtime_error("too many redirects for " + url);
}

std::string get_text(const std::string& path) {
  std::string body;
  const std::string url = path.rfind("http", 0) == 0 ? path : std::string(kHost) + path;
  const int status = http_get(
      url, [&](const char* data, size_t len) { body.append(data, len); return true; }, nullptr);
  if (status != 200)
    throw std::runtime_error("huggingface.co returned HTTP " + std::to_string(status) + " for " +
                             path);
  return body;
}

/** Streams a file to disk. The body is written as it arrives rather than
 *  assembled in memory: the samplers are a quarter of a gigabyte each, and a
 *  server that doubles that in RSS to install a model is not a serving one. */
void download(const std::string& path, const fs::path& dest, const std::string& label,
              const ProgressFn& progress, FetchProgress& state, const std::atomic<bool>& cancel) {
  fs::create_directories(dest.parent_path());
  const fs::path tmp = dest.string() + ".part";
  std::ofstream out(tmp, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + tmp.string());

  state.file = label;
  state.done = 0;
  state.total = 0;
  progress(state);

  uint64_t seen = 0;
  int status = 0;
  try {
    status = http_get(
        std::string(kHost) + path,
        [&](const char* data, size_t len) {
          if (cancel.load(std::memory_order_relaxed)) return false;
          out.write(data, static_cast<std::streamsize>(len));
          seen += len;
          // Roughly every megabyte: often enough for a progress bar, rarely
          // enough that the callback is the expensive part of the download.
          if (seen - state.done >= (1u << 20)) {
            state.done = seen;
            progress(state);
          }
          return out.good();
        },
        [&](uint64_t, uint64_t total) {
          state.total = total;
          return true;
        });
  } catch (...) {
    out.close();
    fs::remove(tmp);
    throw;
  }

  out.close();
  if (cancel.load(std::memory_order_relaxed)) {
    fs::remove(tmp);
    throw std::runtime_error("cancelled");
  }
  if (status != 200 && status != 206) {
    fs::remove(tmp);
    throw std::runtime_error("download failed for " + label + " (HTTP " + std::to_string(status) +
                             ")");
  }
  state.done = seen;
  progress(state);
  fs::rename(tmp, dest);
}

/** Minimal .npy reader for the one case this needs: a C-order float32 array.
 *  Writing a general parser would be more code than the format is worth here,
 *  so anything else is refused by name rather than misread. */
std::vector<float> read_npy(const std::string& blob, std::vector<int64_t>& shape) {
  if (blob.size() < 10 || std::memcmp(blob.data(), "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file");
  const uint8_t major = static_cast<uint8_t>(blob[6]);
  size_t header_len = 0, offset = 0;
  if (major == 1) {
    uint16_t n = 0;
    std::memcpy(&n, blob.data() + 8, 2);
    header_len = n;
    offset = 10;
  } else {
    uint32_t n = 0;
    std::memcpy(&n, blob.data() + 8, 4);
    header_len = n;
    offset = 12;
  }
  const std::string header = blob.substr(offset, header_len);
  if (header.find("'<f4'") == std::string::npos && header.find("\"<f4\"") == std::string::npos)
    throw std::runtime_error("style arrays must be little-endian float32");
  if (header.find("'fortran_order': True") != std::string::npos)
    throw std::runtime_error("style arrays must be C-ordered");

  shape.clear();
  const auto open = header.find("'shape':");
  const auto lb = header.find('(', open);
  const auto rb = header.find(')', lb);
  std::string dims = header.substr(lb + 1, rb - lb - 1);
  for (size_t i = 0; i < dims.size();) {
    while (i < dims.size() && !std::isdigit(static_cast<unsigned char>(dims[i]))) ++i;
    if (i >= dims.size()) break;
    size_t j = i;
    while (j < dims.size() && std::isdigit(static_cast<unsigned char>(dims[j]))) ++j;
    shape.push_back(std::stoll(dims.substr(i, j - i)));
    i = j;
  }

  const size_t start = offset + header_len;
  std::vector<float> out((blob.size() - start) / sizeof(float));
  std::memcpy(out.data(), blob.data() + start, out.size() * sizeof(float));
  return out;
}

/** Supertonic ships its styles as JSON; the arrays come flat with the shape
 *  either alongside or implied by the release's fixed geometry. */
std::vector<float> read_json_style(const json& node, std::vector<int64_t>& shape,
                                   const std::vector<int64_t>& fallback) {
  const json* values = &node;
  if (node.is_object()) {
    for (const char* key : {"data", "values", "array"})
      if (node.contains(key)) values = &node.at(key);
    if (node.contains("shape")) shape = node.at("shape").get<std::vector<int64_t>>();
  }
  std::vector<float> flat;
  std::function<void(const json&)> walk = [&](const json& n) {
    if (n.is_array()) {
      for (const auto& x : n) walk(x);
    } else if (n.is_number()) {
      flat.push_back(n.get<float>());
    }
  };
  walk(*values);
  if (shape.empty()) shape = fallback;
  int64_t want = 1;
  for (int64_t d : shape) want *= d;
  if (want != static_cast<int64_t>(flat.size()))
    throw std::runtime_error("style array has " + std::to_string(flat.size()) +
                             " values but its shape wants " + std::to_string(want));
  return flat;
}

StTensor as_tensor(const std::vector<float>& v, const std::vector<int64_t>& shape) {
  StTensor t;
  t.dtype = "F32";
  t.shape = shape;
  t.data.resize(v.size() * sizeof(float));
  std::memcpy(t.data.data(), v.data(), t.data.size());
  return t;
}

}  // namespace

const std::vector<CatalogueEntry>& fetch_catalogue() {
  static const std::vector<CatalogueEntry> all{
      {"TeraTTSv2", "TeraSpace/TeraTTSv2", "s3", "TeraTTS v2", "ru, en", false, 640ull << 20},
      {"supertonic-3", "Supertone/supertonic-3", "s3", "Supertonic 3", "en, ko, ja", false,
       400ull << 20},
  };
  return all;
}

const CatalogueEntry* find_in_catalogue(const std::string& id_or_repo) {
  for (const auto& e : fetch_catalogue())
    if (e.id == id_or_repo || e.repo == id_or_repo) return &e;
  return nullptr;
}

void fetch_model(const CatalogueEntry& entry, const fs::path& dir, const ProgressFn& progress,
                 const std::atomic<bool>& cancel) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
  (void)entry; (void)dir; (void)progress; (void)cancel;
  throw std::runtime_error(
      "this build has no TLS support, so it cannot download a model; install OpenSSL and rebuild, "
      "or prepare the model with install.sh");
#else
  const Layout& layout = entry.repo.rfind("TeraSpace/", 0) == 0 ? kTera : kSupertonic;

  FetchProgress state;
  state.steps = 7;
  state.step = 1;
  state.stage = "listing";
  progress(state);

  const auto meta = json::parse(get_text("/api/models/" + entry.repo));
  std::vector<std::string> files;
  for (const auto& s : meta.value("siblings", json::array()))
    files.push_back(s.value("rfilename", std::string{}));
  auto has = [&](const std::string& f) {
    for (const auto& x : files)
      if (x == f) return true;
    return false;
  };

  std::string sampler;
  for (const char* candidate : layout.sampler_candidates) {
    const std::string path = std::string(layout.graph_dir) + "/" + candidate;
    if (has(path)) {
      sampler = path;
      break;
    }
  }
  if (sampler.empty()) throw std::runtime_error(entry.repo + " has no sampler graph");

  fs::create_directories(dir);
  const std::string base = "/" + entry.repo + "/resolve/main/";
  const std::vector<std::pair<std::string, std::string>> graphs{
      {"text_encoder.onnx", std::string(layout.graph_dir) + "/text_encoder.onnx"},
      {"duration_predictor.onnx", std::string(layout.graph_dir) + "/duration_predictor.onnx"},
      {"sampler.onnx", sampler},
      {"vocoder.onnx", std::string(layout.graph_dir) + "/vocoder.onnx"},
  };

  state.stage = "downloading";
  for (const auto& [local, remote] : graphs) {
    if (!has(remote)) throw std::runtime_error(entry.repo + " has no " + remote);
    ++state.step;
    download(base + remote, dir / local, local, progress, state, cancel);
  }

  ++state.step;
  state.stage = "character table";
  download(base + layout.indexer, dir / "unicode_indexer.json", "unicode_indexer.json", progress,
           state, cancel);

  ++state.step;
  state.stage = "voices";
  state.file.clear();
  progress(state);

  const fs::path voices = dir / "voices";
  fs::create_directories(voices);
  const std::string prefix = std::string(layout.styles_dir) + "/";
  std::vector<std::string> names;
  for (const auto& f : files) {
    if (f.rfind(prefix, 0) != 0) continue;
    std::string rest = f.substr(prefix.size());
    if (layout.styles_are_npy) {
      const auto slash = rest.find('/');
      if (slash == std::string::npos) continue;
      rest = rest.substr(0, slash);
    } else {
      if (rest.size() < 6 || rest.substr(rest.size() - 5) != ".json") continue;
      rest = rest.substr(0, rest.size() - 5);
    }
    if (std::find(names.begin(), names.end(), rest) == names.end()) names.push_back(rest);
  }
  if (names.empty()) throw std::runtime_error(entry.repo + " ships no voices");

  std::vector<int64_t> ttl_shape{1, 50, 256}, dp_shape{1, 8, 16};
  int made = 0;
  for (const auto& name : names) {
    if (cancel.load(std::memory_order_relaxed)) throw std::runtime_error("cancelled");
    try {
      std::vector<float> ttl, dp;
      if (layout.styles_are_npy) {
        ttl = read_npy(get_text(base + prefix + name + "/style_ttl.npy"), ttl_shape);
        dp = read_npy(get_text(base + prefix + name + "/style_dp.npy"), dp_shape);
      } else {
        const auto blob = json::parse(get_text(base + prefix + name + ".json"));
        auto pick = [&](std::initializer_list<const char*> keys) -> const json& {
          for (const char* k : keys)
            if (blob.contains(k)) return blob.at(k);
          throw std::runtime_error("no style array in " + name + ".json");
        };
        ttl = read_json_style(pick({"style_ttl", "ttl", "style"}), ttl_shape, {1, 50, 256});
        dp = read_json_style(pick({"style_dp", "dp"}), dp_shape, {1, 8, 16});
      }
      st_save(voices / (name + ".safetensors"),
              {{"style_ttl", as_tensor(ttl, ttl_shape)}, {"style_dp", as_tensor(dp, dp_shape)}},
              {{"architecture", "s3"}});
      fs::permissions(voices / (name + ".safetensors"),
                      fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read |
                          fs::perms::others_read);
      ++made;
      state.file = name;
      progress(state);
    } catch (const std::exception& e) {
      // One unreadable voice must not sink an otherwise complete install.
      std::fprintf(stderr, "voice %s skipped: %s\n", name.c_str(), e.what());
    }
  }
  if (made == 0) throw std::runtime_error("no voices could be read; the model would be mute");

  ++state.step;
  state.stage = "manifest";
  state.file.clear();
  progress(state);

  json manifest{
      {"schema_version", 1},
      {"architecture", entry.architecture},
      {"bundle_name", entry.id},
      {"sample_rate", layout.sample_rate},
      {"samples_per_frame", kSamplesPerFrame},
      {"frame_rate", static_cast<double>(layout.sample_rate) / kSamplesPerFrame},
      {"s3",
       {{"latent_dim", 144},
        {"vocoder_context_frames", 20},
        {"stream_chunk_frames", 16},
        {"speed", layout.speed},
        {"guidance", 3.0},
        {"style_ttl_shape", ttl_shape},
        {"style_dp_shape", dp_shape},
        {"language_tags", layout.language_tags},
        {"default_language", layout.default_language}}},
      {"defaults", {{"temperature", 0.5}, {"eos_threshold", -4.0}, {"lsd_steps", 1}}}};
  std::ofstream out(dir / "bundle.json");
  out << manifest.dump(2) << "\n";
  if (!out) throw std::runtime_error("cannot write " + (dir / "bundle.json").string());

  state.step = state.steps;
  state.stage = "ready";
  progress(state);
#endif
}

}  // namespace nanotts
