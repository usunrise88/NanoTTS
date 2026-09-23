#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio/resample.hpp"
#include "audio/wav.hpp"
#include "engine/engine.hpp"
#include "http/server.hpp"
#include "numa/topology.hpp"
#include "text/text.hpp"

namespace fs = std::filesystem;
using namespace xvibe;

namespace {

struct Args {
  std::string command;
  fs::path bundle = "bundle";
  fs::path tokenizer;
  fs::path voice;
  fs::path input;
  fs::path output = "output.wav";
  std::string text;
  bool int8 = false;
  int threads = 4;
  int node = -1;
  float temperature = -1.f;
  float eos_threshold = 2.f;  // sentinel: "use the bundle default"
  int lsd_steps = -1;
  int first_chunk = 2;
  int chunk = 15;
  uint64_t seed = 0;
  int repeat = 1;
  // serve
  std::string host = "0.0.0.0";
  int port = 8080;
  fs::path voices = "voices";
  fs::path web;  // empty -> use the embedded console
  std::string nodes;
  int workers_per_node = 1;
  bool smt = false;
  bool no_pin = false;
  std::string api_key;
  std::string cors_origin = "*";
  std::string accent_url;
};

[[noreturn]] void usage() {
  std::cerr <<
      R"(xvibe-tts - CPU inference for xVibePocketTTS

  xvibe-tts info      --bundle DIR
  xvibe-tts topology
  xvibe-tts generate  --bundle DIR --tokenizer FILE --voice FILE --text STR [--output WAV]
  xvibe-tts warm      --bundle DIR --tokenizer FILE --input REF.wav --output VOICE.safetensors
  xvibe-tts import    --bundle DIR --tokenizer FILE --input UPSTREAM.safetensors --output VOICE.safetensors
  xvibe-tts serve     --bundle DIR --tokenizer FILE --voices DIR [--port N] [--nodes 0,1]

Options:
  --int8              use quantised graphs where available
  --threads N         intra-op threads per engine (default 4)
  --node N            pin to NUMA node N and allocate there
  --temperature F     default: bundle value (0.5)
  --eos-threshold F   default: bundle value (-1.0)
  --lsd-steps N       flow solver steps (default 1)
  --first-chunk N     frames in the first decoded chunk (default 2, lower = better TTFB)
  --chunk N           frames per later chunk (default 15)
  --seed N            0 picks a random seed
  --repeat N          generate N times, for benchmarking

Serve options:
  --host H --port N   listen address (default 0.0.0.0:8080)
  --voices DIR        warmed voice directory (default ./voices)
  --web DIR           serve the console from disk instead of the embedded bundle
  --nodes 0,1,2       NUMA nodes to place workers on (default: all)
  --workers-per-node N  engines per node (default 1)
  --smt               let workers use SMT siblings as well as physical cores
  --no-pin            do not pin threads or bind memory
  --api-key KEY       require 'Authorization: Bearer KEY'
  --cors-origin O     Access-Control-Allow-Origin (default *, "" disables)
  --accent-url URL    RUAccent sidecar; without it callers must supply stress
)";
  std::exit(2);
}

Args parse(int argc, char** argv) {
  if (argc < 2) usage();
  Args a;
  a.command = argv[1];
  auto need = [&](int& i) -> std::string {
    if (++i >= argc) usage();
    return argv[i];
  };
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--bundle") a.bundle = need(i);
    else if (k == "--tokenizer") a.tokenizer = need(i);
    else if (k == "--voice") a.voice = need(i);
    else if (k == "--input") a.input = need(i);
    else if (k == "--output") a.output = need(i);
    else if (k == "--text") a.text = need(i);
    else if (k == "--int8") a.int8 = true;
    else if (k == "--threads") a.threads = std::stoi(need(i));
    else if (k == "--node") a.node = std::stoi(need(i));
    else if (k == "--temperature") a.temperature = std::stof(need(i));
    else if (k == "--eos-threshold") a.eos_threshold = std::stof(need(i));
    else if (k == "--lsd-steps") a.lsd_steps = std::stoi(need(i));
    else if (k == "--first-chunk") a.first_chunk = std::stoi(need(i));
    else if (k == "--chunk") a.chunk = std::stoi(need(i));
    else if (k == "--seed") a.seed = std::stoull(need(i));
    else if (k == "--repeat") a.repeat = std::stoi(need(i));
    else if (k == "--host") a.host = need(i);
    else if (k == "--port") a.port = std::stoi(need(i));
    else if (k == "--voices") a.voices = need(i);
    else if (k == "--web") a.web = need(i);
    else if (k == "--nodes") a.nodes = need(i);
    else if (k == "--workers-per-node") a.workers_per_node = std::stoi(need(i));
    else if (k == "--smt") a.smt = true;
    else if (k == "--no-pin") a.no_pin = true;
    else if (k == "--api-key") a.api_key = need(i);
    else if (k == "--cors-origin") a.cors_origin = need(i);
    else if (k == "--accent-url") a.accent_url = need(i);
    else usage();
  }
  if (a.tokenizer.empty()) a.tokenizer = a.bundle / "tokenizer.model";
  return a;
}

EngineConfig engine_config(const Args& a, const Topology& topo) {
  EngineConfig cfg;
  if (!a.accent_url.empty()) cfg.accent = std::make_shared<Accentuator>(a.accent_url);
  cfg.tokenizer_path = a.tokenizer;
  cfg.int8 = a.int8;
  cfg.threads = a.threads;
  cfg.numa_node = a.node;
  if (a.node >= 0) {
    for (const auto& n : topo.nodes)
      if (n.id == a.node) cfg.cpus = n.cpus;
    if (cfg.cpus.empty()) throw std::runtime_error("no such NUMA node: " + std::to_string(a.node));
  }
  return cfg;
}

GenParams gen_params(const Args& a, const Bundle& b) {
  GenParams p;
  p.temperature = a.temperature >= 0.f ? a.temperature : b.default_temperature;
  p.eos_threshold = a.eos_threshold < 2.f ? a.eos_threshold : b.default_eos_threshold;
  p.lsd_steps = a.lsd_steps > 0 ? a.lsd_steps : b.default_lsd_steps;
  p.first_chunk_frames = a.first_chunk;
  p.chunk_frames = a.chunk;
  p.seed = a.seed;
  return p;
}

// Prints exactly what the text layer hands the model, so it can be diffed
// against the Python reference token for token.
int cmd_chunks(const Args& a) {
  const auto b = Bundle::load(a.bundle);
  Tokenizer tok(a.tokenizer);
  const std::string stressed = to_model_stress(a.text);
  std::cout << "stressed: " << stressed << "\n";
  // What the accent sidecar would be asked for before the first chunk can start.
  const auto [lead, rest] = split_lead(tok, stressed, (b.max_token_per_chunk * 70) / 100,
                                       b.pad_with_spaces, b.remove_semicolons);
  std::cout << "accent lead: " << lead << "\n"
            << "accent tail: " << (rest.empty() ? "(none)" : rest) << "\n";
  const auto chunks = split_chunks(tok, stressed, b.max_token_per_chunk, b.pad_with_spaces,
                                   b.remove_semicolons);
  for (size_t i = 0; i < chunks.size(); ++i) {
    auto [prepared, guess] = prepare_text_prompt(chunks[i], b.pad_with_spaces, b.remove_semicolons);
    const auto ids = tok.encode(prepared);
    std::cout << "chunk " << i << " frames_after_eos=" << (guess + 2) << " prepared=" << prepared
              << "\n  ids=[";
    for (size_t k = 0; k < ids.size(); ++k) std::cout << (k ? "," : "") << ids[k];
    std::cout << "]\n";
  }
  return 0;
}

int cmd_info(const Args& a) {
  const auto b = Bundle::load(a.bundle);
  std::cout << "bundle          " << a.bundle << "\n"
            << "sample rate     " << b.sample_rate << " Hz, " << b.frame_rate << " frames/s ("
            << b.samples_per_frame << " samples/frame)\n"
            << "flow lm         " << b.num_layers << " layers x " << b.num_heads << " heads x "
            << b.dim_per_head << " (d_model " << b.cond_dim << ")\n"
            << "latent dim      " << b.latent_dim << "\n"
            << "max_seq         " << b.max_seq << " positions, mimi cache " << b.mimi_cache_len
            << "\n"
            << "vocab           " << b.vocab_size << "\n"
            << "states          " << b.flow_state.size() << " flow, " << b.mimi_state.size()
            << " mimi\n"
            << "defaults        temperature " << b.default_temperature << ", eos "
            << b.default_eos_threshold << ", lsd steps " << b.default_lsd_steps << "\n"
            << "bos_before_voice" << (b.bos_before_voice.empty() ? " missing" : " present") << "\n";
  return 0;
}

int cmd_generate(const Args& a) {
  const auto topo = Topology::detect();
  auto bundle = Bundle::load(a.bundle);
  Engine engine(bundle, engine_config(a, topo));

  const auto voice = a.voice.extension() == ".safetensors" && fs::exists(a.voice)
                         ? engine.load_voice_file(a.voice)
                         : VoiceState{};
  if (!voice.state) throw std::runtime_error("--voice must point at a warmed .safetensors file");

  const auto params = gen_params(a, bundle);
  for (int r = 0; r < a.repeat; ++r) {
    std::vector<float> pcm;
    engine.reset_timings();
    const auto t0 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_first{};
    bool first = true;

    engine.generate(a.text, voice, params, [&](const float* s, size_t n) {
      if (first) {
        t_first = std::chrono::steady_clock::now();
        first = false;
      }
      pcm.insert(pcm.end(), s, s + n);
      return true;
    });

    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double ttfb = first ? 0.0 : std::chrono::duration<double>(t_first - t0).count();
    const double dur = static_cast<double>(pcm.size()) / bundle.sample_rate;
    std::cout << "run " << (r + 1) << "/" << a.repeat << "  audio " << dur << " s  wall " << wall
              << " s  RTF " << (wall / (dur > 0 ? dur : 1)) << "  (" << (dur / (wall > 0 ? wall : 1))
              << "x realtime)  TTFB " << ttfb * 1000.0 << " ms\n"
              << "        " << engine.timings().describe() << "\n";
    if (r == a.repeat - 1) {
      wav_write(a.output, pcm, bundle.sample_rate);
      std::cout << "wrote " << a.output << "\n";
    }
  }
  return 0;
}

int cmd_warm(const Args& a) {
  const auto topo = Topology::detect();
  auto bundle = Bundle::load(a.bundle);
  Engine engine(bundle, engine_config(a, topo));

  auto audio = wav_read(a.input);
  if (audio.sample_rate != bundle.sample_rate) {
    std::cout << "resampling " << audio.sample_rate << " -> " << bundle.sample_rate << " Hz\n";
    audio.samples = resample(audio.samples, audio.sample_rate, bundle.sample_rate);
  }
  const double seconds = static_cast<double>(audio.samples.size()) / bundle.sample_rate;
  std::cout << "reference " << seconds << " s\n";

  const auto t0 = std::chrono::steady_clock::now();
  const auto voice = engine.warm_voice(audio.samples);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  engine.save_voice_file(a.output, voice);
  std::cout << "warmed in " << wall * 1000.0 << " ms, prefix " << voice.prefix << " frames -> "
            << a.output << "\n";
  return 0;
}

int cmd_serve(const Args& a) {
  ServerConfig cfg;
  cfg.host = a.host;
  cfg.port = a.port;
  cfg.bundle_dir = a.bundle;
  cfg.tokenizer_path = a.tokenizer;
  cfg.voices_dir = a.voices;
  cfg.web_dir = a.web;
  cfg.int8 = a.int8;
  cfg.workers_per_node = a.workers_per_node;
  cfg.threads = a.threads;
  cfg.use_smt = a.smt;
  cfg.pin = !a.no_pin;
  cfg.api_key = a.api_key;
  cfg.cors_origin = a.cors_origin;
  cfg.accent_url = a.accent_url;
  if (!a.nodes.empty()) {
    std::stringstream ss(a.nodes);
    std::string tok;
    while (std::getline(ss, tok, ',')) cfg.nodes.push_back(std::stoi(tok));
  }
  Service svc(cfg, Bundle::load(a.bundle));
  return svc.run();
}

int cmd_import(const Args& a) {
  const auto topo = Topology::detect();
  auto bundle = Bundle::load(a.bundle);
  Engine engine(bundle, engine_config(a, topo));
  const auto voice = engine.import_upstream_voice(a.input);
  engine.save_voice_file(a.output, voice);
  std::cout << "imported prefix " << voice.prefix << " frames -> " << a.output << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.command == "info") return cmd_info(a);
    if (a.command == "topology") {
      std::cout << Topology::detect().describe();
      return 0;
    }
    if (a.command == "chunks") return cmd_chunks(a);
    if (a.command == "generate") return cmd_generate(a);
    if (a.command == "warm") return cmd_warm(a);
    if (a.command == "import") return cmd_import(a);
    if (a.command == "serve") return cmd_serve(a);
    usage();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
