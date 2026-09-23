#include "text/s3_text.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "text/numbers.hpp"
#include "text/text.hpp"

namespace nanotts {
namespace {

using nlohmann::json;

bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool is_letter(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
         (cp >= 0x0410 && cp <= 0x044F) || cp == 0x0401 || cp == 0x0451;
}

bool needs_space_after(uint32_t cp) {
  return cp == ',' || cp == '.' || cp == '!' || cp == '?' || cp == ';' || cp == ':' ||
         cp == 0x2026;  // …
}

/** The compatibility decompositions these vocabularies actually need. The small
 *  Cyrillic tables carry `и` + U+0306 and `е` + U+0308 rather than `й` and `ё`,
 *  so a precomposed letter has to be taken apart or it is dropped as unknown.
 *  Doing only this, rather than pulling in a full normaliser, keeps the
 *  behaviour inspectable: the vocabulary is 134 characters wide. */
bool decompose(uint32_t cp, uint32_t& base, uint32_t& mark) {
  switch (cp) {
    case 0x0439: base = 0x0438; mark = 0x0306; return true;  // й
    case 0x0419: base = 0x0418; mark = 0x0306; return true;  // Й
    case 0x0451: base = 0x0435; mark = 0x0308; return true;  // ё
    case 0x0401: base = 0x0415; mark = 0x0308; return true;  // Ё
    default: return false;
  }
}

/** Separates punctuation from the word that follows, except between the digits
 *  of a decimal literal. */
std::string space_punctuation(const std::string& text) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  out.reserve(cps.size() + 8);
  for (size_t i = 0; i < cps.size(); ++i) {
    out.push_back(cps[i]);
    if (!needs_space_after(cps[i])) continue;
    const uint32_t prev = i ? cps[i - 1] : 0;
    const uint32_t next = i + 1 < cps.size() ? cps[i + 1] : 0;
    if (next == 0 || next == ' ') continue;
    if ((cps[i] == '.' || cps[i] == ',') && is_digit(prev) && is_digit(next)) continue;
    out.push_back(' ');
  }
  return utf8_encode(out);
}

/** A digit running straight into a letter reads as two things, so it is split. */
std::string space_digit_letter(const std::string& text) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  for (size_t i = 0; i < cps.size(); ++i) {
    if (i && is_digit(cps[i - 1]) && is_letter(cps[i])) out.push_back(' ');
    out.push_back(cps[i]);
  }
  return utf8_encode(out);
}

/** Applies the known decompositions and drops whatever the table still has no
 *  token for, recording what went. */
std::string filter(const std::string& text, const UnicodeIndexer& indexer, bool keep_digits,
                   std::string& skipped) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  out.reserve(cps.size());
  for (uint32_t cp : cps) {
    if (indexer.supports(cp)) {
      out.push_back(cp);
      continue;
    }
    uint32_t base = 0, mark = 0;
    if (decompose(cp, base, mark) && indexer.supports(base) && indexer.supports(mark)) {
      out.push_back(base);
      out.push_back(mark);
      continue;
    }
    if (keep_digits && is_digit(cp)) {
      out.push_back(cp);
      continue;
    }
    skipped += utf8_encode({cp});
  }
  return utf8_encode(out);
}

/** Replaces every numeric literal with its spelling. */
std::string expand_numbers(const std::string& text, const std::string& lang) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  size_t i = 0;
  while (i < cps.size()) {
    const bool sign = cps[i] == '-' || cps[i] == 0x2212;
    size_t j = i + (sign ? 1 : 0);
    if (j >= cps.size() || !is_digit(cps[j])) {
      out.push_back(cps[i]);
      ++i;
      continue;
    }
    // A literal is digits, optionally with one decimal separator inside.
    bool seen_sep = false;
    size_t k = j;
    while (k < cps.size()) {
      if (is_digit(cps[k])) { ++k; continue; }
      if (!seen_sep && (cps[k] == '.' || cps[k] == ',') && k + 1 < cps.size() && is_digit(cps[k + 1])) {
        seen_sep = true;
        ++k;
        continue;
      }
      break;
    }
    const std::string literal =
        utf8_encode(std::vector<uint32_t>(cps.begin() + static_cast<long>(i), cps.begin() + static_cast<long>(k)));
    const auto words = spell_number(literal, lang);
    for (uint32_t cp : utf8_decode(words)) out.push_back(cp);
    i = k;
  }
  return utf8_encode(out);
}

bool has_language_tag(const std::string& text) {
  return text.find("<ru>") != std::string::npos || text.find("<en>") != std::string::npos;
}

std::string strip_plus(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text)
    if (c != '+') out += c;
  return out;
}

}  // namespace

UnicodeIndexer::UnicodeIndexer(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  json j;
  in >> j;
  if (!j.is_array()) throw std::runtime_error(path.string() + ": expected an array");
  table_ = j.get<std::vector<int32_t>>();
  if (table_.size() != 65536)
    throw std::runtime_error(path.string() + ": expected 65536 entries, got " +
                             std::to_string(table_.size()));
  digits_ = true;
  for (uint32_t c = '0'; c <= '9'; ++c) digits_ = digits_ && table_[c] >= 0;
}

std::vector<int64_t> UnicodeIndexer::encode(const std::string& text) const {
  std::vector<int64_t> ids;
  for (uint32_t cp : utf8_decode(text)) {
    const int32_t t = token(cp);
    if (t < 0)
      throw std::runtime_error("unsupported codepoint U+" + std::to_string(cp) +
                               " reached the encoder");
    ids.push_back(t);
  }
  if (ids.empty()) throw std::runtime_error("text produced no tokens");
  return ids;
}

S3Text prepare_s3_text(const std::string& input, const UnicodeIndexer& indexer,
                       const S3TextConfig& cfg, Accentuator* accent) {
  S3Text result;
  std::string text = space_punctuation(input);
  text = space_digit_letter(text);
  // Digits survive this pass only so the expansion below can see them.
  text = filter(text, indexer, /*keep_digits=*/true, result.skipped);

  std::string language = cfg.default_language;
  if (cfg.language_tags) {
    // Our API takes plain text like the rest of the service, so text that
    // carries no span gets wrapped rather than rejected.
    if (has_language_tag(text)) {
      language = text.find("<en>") != std::string::npos && text.find("<ru>") == std::string::npos
                     ? "en"
                     : "ru";
    } else {
      text = "<" + language + ">" + text + "</" + language + ">";
    }
  }

  if (cfg.expand_numbers || !indexer.supports_digits()) text = expand_numbers(text, language);
  text = filter(text, indexer, /*keep_digits=*/false, result.skipped);

  // Stress goes on last so the sidecar sees words rather than digits. It speaks
  // the combining-acute notation; s3 wants `+` in front of the vowel.
  if (accent != nullptr && language == "ru") {
    const std::string acute = accent->apply(to_model_stress(text));
    text = cfg.stress_plus ? to_plus_stress(acute) : acute;
    text = filter(text, indexer, /*keep_digits=*/false, result.skipped);
  }

  result.model = text;
  result.duration = cfg.stress_plus ? strip_plus(text) : text;
  return result;
}

}  // namespace nanotts
