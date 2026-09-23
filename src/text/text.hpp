#pragma once
#include <string>
#include <utility>
#include <vector>

#include "model/tokenizer.hpp"

namespace xvibe {

// `+о` -> `о` + U+0301. Marks on ё are removed: ё is inherently stressed, and
// an extra combining acute pushes the tokenizer out of distribution. A `+` that
// is not in front of a Russian vowel stays literal, so "C++" and "2+2" survive.
std::string to_model_stress(const std::string& text);

// Port of pocket_tts prepare_text_prompt. Returns the prepared text and the
// model's frames_after_eos guess. The single-pass "  " -> " " collapse is not
// idempotent; that is upstream behaviour and changing it changes the chunking.
std::pair<std::string, int> prepare_text_prompt(const std::string& text, bool pad_with_spaces,
                                                bool remove_semicolons);

// Sentence split on ".!...?" tokens, comma fallback for oversized sentences,
// then a greedy merge up to max_tokens. Token-based, not character-based.
std::vector<std::string> split_chunks(const Tokenizer& tok, const std::string& text,
                                      int max_tokens, bool pad_with_spaces,
                                      bool remove_semicolons);

// Splits `text` on sentence boundaries into a leading part holding at least
// `min_tokens` tokens and the remainder. Used to accent only as much text as the
// first chunk needs: the cut never lands inside a sentence, so chunking the two
// halves afterwards yields the same boundaries as chunking the whole.
std::pair<std::string, std::string> split_lead(const Tokenizer& tok, const std::string& text,
                                               int min_tokens, bool pad_with_spaces,
                                               bool remove_semicolons);

// UTF-8 helpers used above, exposed for tests.
std::vector<uint32_t> utf8_decode(const std::string& s);
std::string utf8_encode(const std::vector<uint32_t>& cps);
bool cp_is_alnum(uint32_t cp);
bool cp_is_upper(uint32_t cp);
uint32_t cp_to_upper(uint32_t cp);

}  // namespace xvibe
