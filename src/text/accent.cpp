#include "text/accent.hpp"

#include <chrono>
#include <nlohmann/json.hpp>
#include <vector>

#include "httplib.h"
#include "text/text.hpp"

namespace xvibe {
namespace {

using nlohmann::json;
constexpr uint32_t kAcute = 0x0301;

// Deliberately word-shaped. A skip_regex match would make RUAccent swallow the
// spaces around it, so the guard has to look like an ordinary word instead.
constexpr const char* kPlaceholderPrefix = "POCKETTTSMANUALSTRESS";
constexpr const char* kPlaceholderSuffix = "TOKEN";

bool is_word_cp(uint32_t cp) { return cp_is_alnum(cp) || cp == kAcute || cp == '_'; }

/** Replaces every word the caller stressed by hand with a placeholder, so
 *  RUAccent leaves it alone. Returns the masked text and the originals. */
std::string protect(const std::string& text, std::vector<std::string>& saved) {
  const auto cps = utf8_decode(text);
  std::string out;
  size_t i = 0;
  while (i < cps.size()) {
    if (!is_word_cp(cps[i])) {
      out += utf8_encode({cps[i]});
      ++i;
      continue;
    }
    size_t j = i;
    bool manual = false;
    while (j < cps.size() && is_word_cp(cps[j])) {
      if (cps[j] == kAcute) manual = true;
      ++j;
    }
    const std::vector<uint32_t> word(cps.begin() + static_cast<long>(i),
                                     cps.begin() + static_cast<long>(j));
    if (manual) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%04zu", saved.size());
      out += std::string(kPlaceholderPrefix) + buf + kPlaceholderSuffix;
      saved.push_back(utf8_encode(word));
    } else {
      out += utf8_encode(word);
    }
    i = j;
  }
  return out;
}

/** Puts the protected words back. RUAccent may change a placeholder's case, so
 *  the search ignores it; a placeholder that went missing means the response
 *  cannot be trusted and the caller falls back to the original text. */
bool restore(std::string& text, const std::vector<std::string>& saved) {
  auto lower = [](std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
  };
  for (size_t k = 0; k < saved.size(); ++k) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04zu", k);
    const std::string token = std::string(kPlaceholderPrefix) + buf + kPlaceholderSuffix;
    const std::string needle = lower(token);
    const std::string haystack = lower(text);
    const auto pos = haystack.find(needle);
    if (pos == std::string::npos) return false;
    text.replace(pos, needle.size(), saved[k]);
  }
  return true;
}

}  // namespace

Accentuator::Accentuator(std::string base_url, size_t cache_entries, int timeout_ms)
    : base_url_(std::move(base_url)), capacity_(cache_entries), timeout_ms_(timeout_ms) {
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
}

bool Accentuator::cache_get(const std::string& key, std::string& out) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) {
    ++stats_.misses;
    return false;
  }
  lru_.splice(lru_.begin(), lru_, it->second);
  out = it->second->second;
  ++stats_.hits;
  return true;
}

void Accentuator::cache_put(const std::string& key, std::string value) {
  std::lock_guard<std::mutex> lock(mu_);
  if (index_.count(key)) return;
  lru_.emplace_front(key, std::move(value));
  index_[key] = lru_.begin();
  while (index_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

std::string Accentuator::request(const std::string& text) {
  httplib::Client client(base_url_);
  client.set_connection_timeout(0, timeout_ms_ * 1000);
  client.set_read_timeout(timeout_ms_ / 1000, (timeout_ms_ % 1000) * 1000);

  const json body = {{"texts", json::array({text})}, {"skip_regex", "[^\\w\\s]"}};
  auto response = client.Post("/accent", body.dump(), "application/json");
  if (!response || response->status != 200)
    throw std::runtime_error(response ? "accent sidecar HTTP " + std::to_string(response->status)
                                      : "accent sidecar unreachable");
  const auto parsed = json::parse(response->body);
  const auto& texts = parsed.at("texts");
  if (!texts.is_array() || texts.empty()) throw std::runtime_error("accent sidecar returned nothing");
  return texts[0].get<std::string>();
}

std::string Accentuator::apply(const std::string& text) {
  if (base_url_.empty() || text.empty()) return text;

  std::string cached;
  if (cache_get(text, cached)) return cached;

  std::vector<std::string> saved;
  const std::string masked = protect(text, saved);

  const auto started = std::chrono::steady_clock::now();
  std::string accented;
  try {
    accented = request(masked);
    available_.store(true, std::memory_order_relaxed);
  } catch (const std::exception&) {
    available_.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.failures;
    return text;  // unaccented beats no audio
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stats_.last_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
  }

  if (!restore(accented, saved)) {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.failures;
    return text;
  }

  // RUAccent answers in `+vowel` notation; the model wants U+0301.
  accented = to_model_stress(accented);
  cache_put(text, accented);
  return accented;
}

bool Accentuator::probe() {
  if (base_url_.empty()) return false;
  httplib::Client client(base_url_);
  client.set_connection_timeout(0, timeout_ms_ * 1000);
  client.set_read_timeout(timeout_ms_ / 1000, (timeout_ms_ % 1000) * 1000);
  auto response = client.Get("/healthz");
  const bool ok = response && response->status == 200;
  available_.store(ok, std::memory_order_relaxed);
  return ok;
}

Accentuator::Stats Accentuator::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

}  // namespace xvibe
