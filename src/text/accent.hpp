#pragma once
#include <atomic>
#include <cstddef>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xvibe {

/** Client for the RUAccent sidecar.
 *
 *  The checkpoint was trained on text carrying U+0301, and synthesising without
 *  it costs 17.7% WER against 3.2% on long sentences, so this is not an
 *  embellishment — it is the difference between the model reading the text and
 *  guessing at it.
 *
 *  Stress the caller wrote by hand always wins. Those words are swapped for
 *  placeholders before the request and put back afterwards, because RUAccent
 *  normalises U+0301 away and re-derives the word otherwise.
 *
 *  A sidecar that is down degrades to unaccented text rather than failing the
 *  request: quieter quality beats a 502.
 */
class Accentuator {
 public:
  Accentuator(std::string base_url, size_t cache_entries = 2048, int timeout_ms = 5000);

  /** `text` must already be in U+0301 notation (run `to_model_stress` first).
   *  Returns accented text; on any failure returns the input unchanged. */
  std::string apply(const std::string& text);

  bool probe();  // one-shot health check, updates `available`
  bool available() const { return available_.load(std::memory_order_relaxed); }
  const std::string& url() const { return base_url_; }

  struct Stats {
    uint64_t hits = 0, misses = 0, failures = 0;
    double last_ms = 0;
  };
  Stats stats() const;

 private:
  std::string request(const std::string& text);
  bool cache_get(const std::string& key, std::string& out);
  void cache_put(const std::string& key, std::string value);

  std::string base_url_;
  size_t capacity_;
  int timeout_ms_;
  std::atomic<bool> available_{false};

  mutable std::mutex mu_;
  std::list<std::pair<std::string, std::string>> lru_;
  std::unordered_map<std::string, decltype(lru_)::iterator> index_;
  Stats stats_;
};

}  // namespace xvibe
