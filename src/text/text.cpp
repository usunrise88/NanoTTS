#include "text/text.hpp"

#include "text/numbers.hpp"

#include <algorithm>
#include <set>
#include <sstream>

namespace nanotts {
namespace {

constexpr uint32_t kAcute = 0x0301;

bool is_russian_vowel(uint32_t cp) {
  static const std::u32string kVowels = U"аеёиоуыэюяАЕЁИОУЫЭЮЯ";
  return kVowels.find(static_cast<char32_t>(cp)) != std::u32string::npos;
}

bool is_yo(uint32_t cp) { return cp == 0x451 || cp == 0x401; }

bool is_space(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}

}  // namespace

std::vector<uint32_t> utf8_decode(const std::string& s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = c;
    size_t extra = 0;
    if (c >= 0xF0) { cp = c & 0x07u; extra = 3; }
    else if (c >= 0xE0) { cp = c & 0x0Fu; extra = 2; }
    else if (c >= 0xC0) { cp = c & 0x1Fu; extra = 1; }
    if (i + extra >= s.size()) extra = 0;  // malformed tail: pass the byte through
    for (size_t k = 1; k <= extra; ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
    out.push_back(cp);
    i += extra + 1;
  }
  return out;
}

std::string utf8_encode(const std::vector<uint32_t>& cps) {
  std::string out;
  out.reserve(cps.size() * 2);
  for (uint32_t cp : cps) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return out;
}

bool cp_is_alnum(uint32_t cp) {
  if (cp < 128) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
  if (cp >= 0x410 && cp <= 0x44F) return true;  // А-Я а-я
  if (is_yo(cp)) return true;
  if (cp >= 0xC0 && cp <= 0x24F) return true;  // Latin-1 supplement / extended
  return false;
}

bool cp_is_upper(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return true;
  if (cp >= 0x410 && cp <= 0x42F) return true;
  return cp == 0x401;
}

uint32_t cp_to_upper(uint32_t cp) {
  if (cp >= 'a' && cp <= 'z') return cp - 32;
  if (cp >= 0x430 && cp <= 0x44F) return cp - 32;
  if (cp == 0x451) return 0x401;
  return cp;
}

std::string to_model_stress(const std::string& text) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  out.reserve(cps.size());
  for (size_t i = 0; i < cps.size(); ++i) {
    const uint32_t cp = cps[i];
    if (cp == '+' && i + 1 < cps.size()) {
      const uint32_t next = cps[i + 1];
      if (is_yo(next)) {  // "+ё" -> "ё": the mark would be redundant
        out.push_back(next);
        ++i;
        continue;
      }
      if (is_russian_vowel(next)) {
        out.push_back(next);
        out.push_back(kAcute);
        ++i;
        continue;
      }
      out.push_back(cp);  // literal plus
      continue;
    }
    if (cp == kAcute && !out.empty() && is_yo(out.back())) continue;  // drop "ё" + acute
    out.push_back(cp);
  }
  return utf8_encode(out);
}

namespace {

std::string strip(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && is_space(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && is_space(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

size_t word_count(const std::string& s) {
  std::istringstream is(s);
  std::string w;
  size_t n = 0;
  while (is >> w) ++n;
  return n;
}

// Python's str.replace scans left to right without re-examining what it wrote,
// so "a   b" becomes "a  b", not "a b". Reproduce that exactly.
std::string replace_double_space_once(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (i + 1 < s.size() && s[i] == ' ' && s[i + 1] == ' ') {
      out.push_back(' ');
      i += 2;
    } else {
      out.push_back(s[i]);
      ++i;
    }
  }
  return out;
}

}  // namespace

std::string to_plus_stress(const std::string& text) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  out.reserve(cps.size());
  for (size_t i = 0; i < cps.size(); ++i) {
    // A combining acute always follows the vowel it marks, so the `+` goes in
    // front of the codepoint already written.
    if (cps[i] == 0x0301 && !out.empty()) {
      const uint32_t vowel = out.back();
      out.pop_back();
      out.push_back('+');
      out.push_back(vowel);
      continue;
    }
    out.push_back(cps[i]);
  }
  return utf8_encode(out);
}

namespace {

/** Characters the Russian checkpoint has no token for, and what to read them
 *  as. The quote mappings keep the pair asymmetric where the source is, so a
 *  parenthetical still opens and closes; the dashes all collapse onto the one
 *  the model knows. */
struct Rewrite {
  uint32_t from;
  const char* to;  // empty means "drop"
};
const Rewrite kRewrites[] = {
    {0x0022, "\xc2\xab"},      // "  -> «   (closing is fixed up below)
    {0x201C, "\xc2\xab"},      // “
    {0x201E, "\xc2\xab"},      // „
    {0x2018, "'"},              // ‘
    {0x201D, "\xc2\xbb"},      // ”
    {0x2019, "'"},              // ’
    {0x201A, ","},              // ‚
    {0x2012, "\xe2\x80\x94"},  // ‒ -> —
    {0x2013, "\xe2\x80\x94"},  // – -> —
    {0x2015, "\xe2\x80\x94"},  // ― -> —
    {0x2212, "-"},              // −
    {0x0028, ","},              // (  a parenthetical reads as a comma
    {0x0029, ","},              // )
    {0x005B, ","},              // [
    {0x005D, ","},              // ]
    {0x007B, ","},              // {
    {0x007D, ","},              // }
    {0x002F, " "},              // /
    {0x005C, " "},              // backslash
    {0x007C, " "},              // |
};

const char* rewrite_for(uint32_t cp) {
  for (const auto& r : kRewrites)
    if (r.from == cp) return r.to;
  return nullptr;
}

}  // namespace

namespace {
bool is_digit_cp(uint32_t cp) { return cp >= '0' && cp <= '9'; }
}  // namespace

std::string spell_numbers(const std::string& text, const std::string& lang) {
  const auto cps = utf8_decode(text);
  std::vector<uint32_t> out;
  size_t i = 0;
  while (i < cps.size()) {
    const bool sign = cps[i] == '-' || cps[i] == 0x2212;
    size_t j = i + (sign ? 1 : 0);
    if (j >= cps.size() || !is_digit_cp(cps[j])) {
      out.push_back(cps[i]);
      ++i;
      continue;
    }
    // A literal is digits, optionally with one decimal separator inside.
    bool seen_sep = false;
    size_t k = j;
    while (k < cps.size()) {
      if (is_digit_cp(cps[k])) { ++k; continue; }
      if (!seen_sep && (cps[k] == '.' || cps[k] == ',') && k + 1 < cps.size() && is_digit_cp(cps[k + 1])) {
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

std::string map_to_vocabulary(const std::string& text, const Tokenizer& tok,
                              std::string& removed) {
  const auto cps = utf8_decode(text);
  std::string out;
  // A straight quote is a single character doing two jobs; alternate it so the
  // pair reads as an opening and a closing one.
  bool quote_open = true;
  for (uint32_t cp : cps) {
    const std::string one = utf8_encode({cp});
    if (!tok.is_unknown(one)) {
      if (cp == 0x00AB) quote_open = false;
      if (cp == 0x00BB) quote_open = true;
      out += one;
      continue;
    }
    const char* to = rewrite_for(cp);
    if (to == nullptr) {
      removed += one;
      continue;
    }
    if (cp == 0x0022) {
      out += quote_open ? "\xc2\xab" : "\xc2\xbb";
      quote_open = !quote_open;
      continue;
    }
    out += to;
  }
  // A bracket that became a comma next to real punctuation leaves ",." behind;
  // the stronger mark wins.
  std::string tidy;
  const auto mapped = utf8_decode(out);
  for (size_t i = 0; i < mapped.size(); ++i) {
    const uint32_t cp = mapped[i];
    if (cp == ',') {
      size_t j = i + 1;
      while (j < mapped.size() && mapped[j] == ' ') ++j;
      if (j < mapped.size() && (mapped[j] == '.' || mapped[j] == ',' || mapped[j] == '!' ||
                                mapped[j] == '?' || mapped[j] == ';' || mapped[j] == ':'))
        continue;
    }
    tidy += utf8_encode({cp});
  }
  return tidy;
}

std::pair<std::string, int> prepare_text_prompt(const std::string& input, bool pad_with_spaces,
                                                bool remove_semicolons) {
  std::string text = strip(input);
  if (text.empty()) throw std::runtime_error("text prompt cannot be empty");

  for (char& c : text)
    if (c == '\n' || c == '\r') c = ' ';
  text = replace_double_space_once(text);
  if (remove_semicolons) std::replace(text.begin(), text.end(), ';', ',');

  const int frames_after_eos = word_count(text) <= 4 ? 3 : 1;

  auto cps = utf8_decode(text);
  if (!cps.empty() && !cp_is_upper(cps[0])) cps[0] = cp_to_upper(cps[0]);
  if (!cps.empty() && cp_is_alnum(cps.back())) cps.push_back('.');
  text = utf8_encode(cps);

  if (pad_with_spaces && word_count(text) < 5) text = std::string(8, ' ') + text;
  return {text, frames_after_eos};
}

namespace {

std::vector<size_t> boundary_indices(const std::vector<int>& tokens,
                                     const std::set<int>& boundary) {
  std::vector<size_t> idx{0};
  bool prev = false;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (boundary.count(tokens[i])) {
      prev = true;
    } else {
      if (prev) idx.push_back(i);
      prev = false;
    }
  }
  idx.push_back(tokens.size());
  return idx;
}

std::vector<std::pair<size_t, std::string>> segments_from(const Tokenizer& tok,
                                                          const std::vector<int>& tokens,
                                                          const std::vector<size_t>& bounds) {
  std::vector<std::pair<size_t, std::string>> out;
  for (size_t i = 0; i + 1 < bounds.size(); ++i) {
    const std::vector<int> slice(tokens.begin() + static_cast<long>(bounds[i]),
                                 tokens.begin() + static_cast<long>(bounds[i + 1]));
    out.emplace_back(bounds[i + 1] - bounds[i], tok.decode(slice));
  }
  return out;
}

// The first token of an encoded punctuation run is SentencePiece's leading
// "▁"; upstream drops it, and the Russian model behaves the same way.
std::set<int> boundary_tokens(const Tokenizer& tok, const std::string& punct) {
  auto ids = tok.encode(punct);
  std::set<int> out;
  for (size_t i = 1; i < ids.size(); ++i) out.insert(ids[i]);
  return out;
}

}  // namespace

std::pair<std::string, std::string> split_lead(const Tokenizer& tok, const std::string& text,
                                               int min_tokens, bool pad_with_spaces,
                                               bool remove_semicolons) {
  auto [prepared, _] = prepare_text_prompt(text, pad_with_spaces, remove_semicolons);
  prepared = strip(prepared);

  const auto tokens = tok.encode(prepared);
  const auto segs = segments_from(tok, tokens, boundary_indices(tokens, boundary_tokens(tok, ".!...?")));

  std::string head, tail;
  size_t head_n = 0;
  for (const auto& [n, seg] : segs) {
    const std::string piece = strip(seg);
    if (piece.empty()) continue;
    // Whole sentences only, and always at least one: a lead that stops mid
    // sentence would cost the accentuator the context it resolves homographs
    // with, and would put a chunk boundary where the prosody does not want one.
    std::string& dst = (head.empty() || static_cast<int>(head_n) < min_tokens) ? head : tail;
    if (!dst.empty()) dst += " ";
    dst += piece;
    if (&dst == &head) head_n += n;
  }
  return {head, tail};
}

std::vector<std::string> split_chunks(const Tokenizer& tok, const std::string& text,
                                      int max_tokens, bool pad_with_spaces,
                                      bool remove_semicolons) {
  auto [prepared, _] = prepare_text_prompt(text, pad_with_spaces, remove_semicolons);
  prepared = strip(prepared);

  const auto tokens = tok.encode(prepared);
  auto segs = segments_from(tok, tokens, boundary_indices(tokens, boundary_tokens(tok, ".!...?")));

  const auto fallback = boundary_tokens(tok, ",;:");
  std::vector<std::pair<size_t, std::string>> refined;
  for (auto& [n, seg] : segs) {
    if (static_cast<int>(n) <= max_tokens) {
      refined.emplace_back(n, seg);
      continue;
    }
    const auto sub = tok.encode(strip(seg));
    auto parts = segments_from(tok, sub, boundary_indices(sub, fallback));
    if (parts.size() > 1)
      refined.insert(refined.end(), parts.begin(), parts.end());
    else
      refined.emplace_back(n, seg);
  }

  std::vector<std::string> chunks;
  std::string cur;
  size_t cur_n = 0;
  for (auto& [n, seg] : refined) {
    if (cur.empty()) {
      cur = seg;
      cur_n = n;
    } else if (static_cast<int>(cur_n + n) > max_tokens) {
      chunks.push_back(strip(cur));
      cur = seg;
      cur_n = n;
    } else {
      cur += " " + seg;
      cur_n += n;
    }
  }
  if (!cur.empty()) chunks.push_back(strip(cur));
  return chunks;
}

}  // namespace nanotts
