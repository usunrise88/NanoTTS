#include "model/bundle.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace xvibe {
namespace fs = std::filesystem;
using nlohmann::json;

size_t StateSpec::element_size() const {
  if (dtype == "float32" || dtype == "int32") return 4;
  if (dtype == "float16") return 2;
  if (dtype == "int64") return 8;
  if (dtype == "bool") return 1;
  throw std::runtime_error("unsupported state dtype: " + dtype);
}

int64_t StateSpec::elements(int64_t batch) const {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) n *= (i == 0 ? batch : shape[i]);
  return n;
}

static std::vector<int64_t> parse_shape(const json& j) {
  std::vector<int64_t> out;
  for (const auto& d : j) out.push_back(d.is_string() ? -1 : d.get<int64_t>());
  return out;
}

static std::vector<StateSpec> parse_states(const json& arr) {
  std::vector<StateSpec> out;
  for (const auto& e : arr) {
    StateSpec s;
    s.name = e.at("name").get<std::string>();
    s.path = e.value("path", s.name);
    s.dtype = e.at("dtype").get<std::string>();
    s.shape = parse_shape(e.at("shape"));
    out.push_back(std::move(s));
  }
  return out;
}

Bundle Bundle::load(const fs::path& dir) {
  std::ifstream in(dir / "bundle.json");
  if (!in) throw std::runtime_error("cannot open " + (dir / "bundle.json").string());
  json j;
  in >> j;

  Bundle b;
  b.dir = dir;
  b.sample_rate = j.value("sample_rate", b.sample_rate);
  b.samples_per_frame = j.value("samples_per_frame", b.samples_per_frame);
  b.frame_rate = j.value("frame_rate", b.frame_rate);
  b.latent_dim = j.value("latent_dim", b.latent_dim);
  b.cond_dim = j.value("conditioning_dim", b.cond_dim);
  b.num_layers = j.value("num_layers", b.num_layers);
  b.num_heads = j.value("num_heads", b.num_heads);
  b.dim_per_head = j.value("dim_per_head", b.dim_per_head);
  b.max_seq = j.value("max_seq", b.max_seq);
  b.mimi_cache_len = j.value("mimi_cache_len", b.mimi_cache_len);
  b.vocab_size = j.value("vocab_size", b.vocab_size);
  b.max_token_per_chunk = j.value("max_token_per_chunk", b.max_token_per_chunk);
  b.insert_bos_before_voice = j.value("insert_bos_before_voice", true);
  b.pad_with_spaces = j.value("pad_with_spaces_for_short_inputs", false);
  b.remove_semicolons = j.value("remove_semicolons", false);

  if (j.contains("defaults")) {
    const auto& d = j.at("defaults");
    b.default_temperature = d.value("temperature", b.default_temperature);
    b.default_eos_threshold = d.value("eos_threshold", b.default_eos_threshold);
    b.default_lsd_steps = d.value("lsd_steps", b.default_lsd_steps);
  }

  b.flow_state = parse_states(j.at("flow_lm_state"));
  b.mimi_state = parse_states(j.at("mimi_state"));

  // bos_before_voice.npy: a tiny, fixed-layout float32 array. Parsing the numpy
  // header properly is overkill for one 1x1xD tensor, so we take the payload
  // after the header length field and check the size matches.
  fs::path bos = dir / "bos_before_voice.npy";
  if (fs::exists(bos)) {
    std::ifstream f(bos, std::ios::binary);
    char magic[6];
    f.read(magic, 6);
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
      uint16_t hl = 0;
      f.read(reinterpret_cast<char*>(&hl), 2);
      header_len = hl;
    } else {
      f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    f.seekg(header_len, std::ios::cur);
    b.bos_before_voice.resize(static_cast<size_t>(b.cond_dim));
    f.read(reinterpret_cast<char*>(b.bos_before_voice.data()),
           static_cast<std::streamsize>(b.bos_before_voice.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated bos_before_voice.npy");
  }
  return b;
}

fs::path Bundle::graph(const std::string& name, bool int8) const {
  if (int8) {
    fs::path q = dir / (name + "_int8.onnx");
    if (fs::exists(q)) return q;
  }
  return dir / (name + ".onnx");
}

}  // namespace xvibe
