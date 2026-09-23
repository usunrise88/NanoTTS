#include <sstream>
#include <stdexcept>

#include "engine/engine.hpp"
#include "engine/s3.hpp"

namespace nanotts {

std::string affinity_string(const std::vector<int>& cpus, int threads) {
  // ORT wants one entry per worker thread, and the calling thread is not one
  // of them, hence threads-1 entries.
  if (cpus.empty() || threads <= 1) return {};
  std::ostringstream os;
  for (int t = 1; t < threads; ++t) {
    if (t > 1) os << ';';
    os << cpus[static_cast<size_t>(t) % cpus.size()];
  }
  return os.str();
}


std::unique_ptr<Backend> make_backend(Bundle bundle, EngineConfig cfg) {
  const std::string arch = bundle.architecture;
  if (arch == "pocket") return std::make_unique<Engine>(std::move(bundle), std::move(cfg));
  if (arch == "s3") return std::make_unique<S3Backend>(std::move(bundle), std::move(cfg));
  throw std::runtime_error("bundle asks for architecture \"" + arch +
                           "\", which this build does not implement; known: pocket, s3");
}

std::string Timings::describe() const {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(1);
  // Only the stages this architecture actually ran; a line of zeroes for the
  // half that does not apply reads as if something went wrong.
  if (accent > 0 || accent_hidden > 0)
    os << "accent " << accent << " (+" << accent_hidden << " hidden)  ";
  if (voice_copy > 0) os << "voice_copy " << voice_copy << "  ";
  if (text_cond > 0) os << "text " << text_cond << "  ";
  if (duration > 0) os << "duration " << duration << "  ";
  if (prefill > 0) os << "prefill " << prefill << "  ";
  if (ar_main > 0) os << "ar_main " << ar_main << "  ";
  if (flow > 0) os << "flow " << flow << "  ";
  if (sampler > 0) os << "sampler " << sampler << "  ";
  if (decode > 0) os << "decode " << decode << "  ";
  os << "(" << frames << " frames, " << chunks << " chunks)";
  return os.str();
}

}  // namespace nanotts
