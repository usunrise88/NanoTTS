#include "text/numbers.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace nanotts {
namespace {

// --- Russian -------------------------------------------------------------
// Masculine nominative, except inside a thousands group, where num2words uses
// the feminine "одна"/"две" because тысяча is feminine.

const char* kRuOnes[20] = {"",          "один",     "два",       "три",       "четыре",
                           "пять",      "шесть",    "семь",      "восемь",    "девять",
                           "десять",    "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
                           "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"};
const char* kRuTens[10] = {"",        "",        "двадцать",   "тридцать", "сорок",
                           "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"};
const char* kRuHundreds[10] = {"",      "сто",      "двести",  "триста",   "четыреста",
                               "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"};

// Scale words in the three forms Russian grammar picks between: 1, 2-4, and the
// rest. The thousands row is feminine, which is what drives the "одна"/"две".
struct RuScale {
  const char* one;
  const char* few;
  const char* many;
  bool feminine;
};
const RuScale kRuScales[] = {
    {"", "", "", false},
    {"тысяча", "тысячи", "тысяч", true},
    {"миллион", "миллиона", "миллионов", false},
    {"миллиард", "миллиарда", "миллиардов", false},
    {"триллион", "триллиона", "триллионов", false},
};

void push(std::string& out, const std::string& word) {
  if (word.empty()) return;
  if (!out.empty()) out += ' ';
  out += word;
}

/** One group of three digits, in words. */
void ru_group(int value, bool feminine, std::string& out) {
  push(out, kRuHundreds[value / 100]);
  const int rest = value % 100;
  if (rest >= 20) {
    push(out, kRuTens[rest / 10]);
    const int ones = rest % 10;
    if (ones) push(out, feminine && ones <= 2 ? (ones == 1 ? "одна" : "две") : kRuOnes[ones]);
  } else if (rest) {
    push(out, feminine && rest <= 2 ? (rest == 1 ? "одна" : "две") : kRuOnes[rest]);
  }
}

/** Which of the three scale forms a count takes. */
const char* ru_scale_word(int64_t group, const RuScale& scale) {
  const int64_t last_two = group % 100;
  const int64_t last = group % 10;
  if (last_two >= 11 && last_two <= 14) return scale.many;
  if (last == 1) return scale.one;
  if (last >= 2 && last <= 4) return scale.few;
  return scale.many;
}

std::string ru_integer(uint64_t value) {
  if (value == 0) return "ноль";
  std::vector<int> groups;  // least significant first
  for (uint64_t v = value; v; v /= 1000) groups.push_back(static_cast<int>(v % 1000));

  std::string out;
  for (size_t i = groups.size(); i-- > 0;) {
    if (groups[i] == 0) continue;
    const RuScale& scale = kRuScales[i < std::size(kRuScales) ? i : std::size(kRuScales) - 1];
    ru_group(groups[i], scale.feminine, out);
    if (i > 0) push(out, ru_scale_word(groups[i], scale));
  }
  return out;
}

// --- English -------------------------------------------------------------

const char* kEnOnes[20] = {"zero",    "one",     "two",       "three",    "four",
                           "five",    "six",     "seven",     "eight",    "nine",
                           "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
                           "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char* kEnTens[10] = {"",      "",      "twenty",  "thirty", "forty",
                           "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* kEnScales[] = {"", "thousand", "million", "billion", "trillion"};

/** One group of three digits. num2words hyphenates the tens and puts "and"
 *  after a hundred when anything follows it. */
std::string en_group(int value) {
  std::string out;
  if (value >= 100) {
    out = std::string(kEnOnes[value / 100]) + " hundred";
    if (value % 100) out += " and ";
  }
  const int rest = value % 100;
  if (rest >= 20) {
    out += kEnTens[rest / 10];
    if (rest % 10) out += std::string("-") + kEnOnes[rest % 10];
  } else if (rest || value == 0) {
    out += kEnOnes[rest];
  }
  return out;
}

std::string en_integer(uint64_t value) {
  if (value == 0) return "zero";
  std::vector<int> groups;
  for (uint64_t v = value; v; v /= 1000) groups.push_back(static_cast<int>(v % 1000));

  std::vector<std::string> parts;
  for (size_t i = groups.size(); i-- > 0;) {
    if (groups[i] == 0) continue;
    std::string part = en_group(groups[i]);
    if (i > 0) part += std::string(" ") + kEnScales[i < std::size(kEnScales) ? i : std::size(kEnScales) - 1];
    parts.push_back(std::move(part));
  }

  // Groups are comma separated, except that a bare tens-or-units tail joins the
  // one before it with "and": "one thousand and one", not "one thousand, one".
  std::string out = parts[0];
  for (size_t i = 1; i < parts.size(); ++i) {
    const bool tail = i + 1 == parts.size() && groups[0] > 0 && groups[0] < 100;
    out += tail ? " and " : ", ";
    out += parts[i];
  }
  return out;
}

// --- fractions -----------------------------------------------------------

const char* kRuFracScale[] = {"",          "десятых",      "сотых",        "тысячных",
                              "десятитысячных", "стотысячных", "миллионных"};

/** num2words reads a decimal as "<whole> целых <fraction> <scale>", with the
 *  feminine agreement that целая/целых takes. */
std::string ru_decimal(const std::string& whole, const std::string& frac) {
  const uint64_t w = std::strtoull(whole.c_str(), nullptr, 10);
  const uint64_t f = std::strtoull(frac.c_str(), nullptr, 10);
  std::string out;
  // "одна целая" / "две целых": the integer part agrees as a feminine noun.
  if (w == 1) out = "одна целая";
  else if (w == 2) out = "две целых";
  else out = ru_integer(w) + " целых";

  std::string frac_words;
  const uint64_t last_two = f % 100, last = f % 10;
  if (f == 1) frac_words = "одна";
  else if (f == 2) frac_words = "две";
  else frac_words = ru_integer(f);
  // Feminine one/two inside the fraction as well.
  if (f > 2 && last == 1 && last_two != 11) {
    const auto pos = frac_words.rfind("один");
    if (pos != std::string::npos) frac_words.replace(pos, std::string("один").size(), "одна");
  } else if (f > 2 && last == 2 && last_two != 12) {
    const auto pos = frac_words.rfind("два");
    if (pos != std::string::npos) frac_words.replace(pos, std::string("два").size(), "две");
  }

  const size_t digits = frac.size();
  const char* scale = digits < std::size(kRuFracScale) ? kRuFracScale[digits] : kRuFracScale[std::size(kRuFracScale) - 1];
  return out + " " + frac_words + " " + scale;
}

/** English reads the fraction digit by digit after "point". */
std::string en_decimal(const std::string& whole, const std::string& frac) {
  std::string out = en_integer(std::strtoull(whole.c_str(), nullptr, 10)) + " point";
  for (char c : frac) {
    if (c < '0' || c > '9') continue;
    out += ' ';
    out += kEnOnes[c - '0'];
  }
  return out;
}

}  // namespace

std::string spell_number(const std::string& literal, const std::string& lang) {
  if (lang != "ru" && lang != "en") return literal;

  std::string text = literal;
  // U+2212 MINUS SIGN is what a typographer's text carries; treat it as a sign.
  const std::string minus_sign = "−";
  for (size_t p = text.find(minus_sign); p != std::string::npos; p = text.find(minus_sign))
    text.replace(p, minus_sign.size(), "-");

  bool negative = false;
  size_t i = 0;
  if (i < text.size() && (text[i] == '-' || text[i] == '+')) {
    negative = text[i] == '-';
    ++i;
  }

  std::string whole, frac;
  bool in_frac = false;
  for (; i < text.size(); ++i) {
    const char c = text[i];
    if (c >= '0' && c <= '9') {
      (in_frac ? frac : whole) += c;
    } else if ((c == '.' || c == ',') && !in_frac) {
      in_frac = true;
    } else {
      return literal;  // not a plain numeric literal; leave it alone
    }
  }
  if (whole.empty() && frac.empty()) return literal;
  if (whole.empty()) whole = "0";

  // Trailing zeros are significant to the reading ("1.50" is fifty hundredths),
  // so the fraction is kept exactly as written.
  std::string words;
  if (in_frac && !frac.empty())
    words = lang == "ru" ? ru_decimal(whole, frac) : en_decimal(whole, frac);
  else
    words = lang == "ru" ? ru_integer(std::strtoull(whole.c_str(), nullptr, 10))
                         : en_integer(std::strtoull(whole.c_str(), nullptr, 10));

  if (negative) words = (lang == "ru" ? "минус " : "minus ") + words;
  return words;
}

}  // namespace nanotts
