#pragma once
#include <string>

namespace nanotts {

/** Spells a numeric literal out in words.
 *
 *  The s3 text encoders were trained on a small character vocabulary, and
 *  TeraTTS v2's has no digits at all -- 134 characters, none of them 0-9 -- so
 *  a number that reaches the model unexpanded is not mispronounced, it is
 *  silently dropped along with whatever it meant. Expansion is therefore part
 *  of the text layer rather than a nicety.
 *
 *  Output matches num2words, which is what the reference implementation uses,
 *  including its choices: Russian cardinals in masculine nominative with the
 *  feminine forms for thousands ("одна тысяча", "две тысячи"), and English with
 *  the "and" after hundreds and commas between groups.
 *
 *  `lang` is "ru" or "en"; anything else returns the literal unchanged, which
 *  keeps a checkpoint whose vocabulary does have digits working.
 */
std::string spell_number(const std::string& literal, const std::string& lang);

}  // namespace nanotts
