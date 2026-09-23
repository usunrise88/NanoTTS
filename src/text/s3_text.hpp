#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "text/accent.hpp"

namespace nanotts {

/** Codepoint -> token table for the s3 family's text encoders.
 *
 *  The released table is a flat array of 65,536 entries, one per BMP codepoint,
 *  with -1 for anything the model was not trained on. The vocabularies are
 *  small and very different between checkpoints: TeraTTS v2 supports 134
 *  characters and no digits at all, Supertonic 3 supports 8,321 including
 *  digits, so what the text layer has to do is a property of the table rather
 *  than of the architecture.
 */
class UnicodeIndexer {
 public:
  explicit UnicodeIndexer(const std::filesystem::path& path);

  /** Token for a codepoint, or -1 when the model has never seen it. */
  int32_t token(uint32_t codepoint) const {
    return codepoint < table_.size() ? table_[codepoint] : -1;
  }
  bool supports(uint32_t codepoint) const { return token(codepoint) >= 0; }
  bool supports_digits() const { return digits_; }

  /** Encodes prepared text. Throws on an unsupported codepoint: by this point
   *  the text layer has already dropped those, so one here is a bug rather
   *  than bad input. */
  std::vector<int64_t> encode(const std::string& text) const;

 private:
  std::vector<int32_t> table_;
  bool digits_ = false;
};

/** How a particular checkpoint wants its text. Read from the bundle, because
 *  these differ between models sharing the same graphs. */
struct S3TextConfig {
  /** TeraTTS v2 was trained with literal <ru>...</ru> spans and refuses text
   *  without them; Supertonic 3 has no language embedding and wants none. */
  bool language_tags = false;
  std::string default_language = "ru";
  /** True when the vocabulary has no digits, so numbers must become words or
   *  be dropped along with their meaning. */
  bool expand_numbers = false;
  /** Stress written as `+` before the vowel. The s3 text encoders use this
   *  spelling where Pocket TTS uses a combining acute. */
  bool stress_plus = true;
};

/** The two strings the graphs want: `model` carries stress marks and goes to
 *  the text encoder, `duration` has them stripped and goes to the duration
 *  predictor, which was trained without them. */
struct S3Text {
  std::string model;
  std::string duration;
  /** Characters dropped because the checkpoint has no token for them. Worth
   *  reporting: silently swallowing half a sentence is the failure mode here. */
  std::string skipped;
};

/** Runs the full text path: punctuation spacing, unsupported-character
 *  filtering, language tags, number expansion, stress, and the compatibility
 *  decomposition the small Cyrillic vocabularies need (they carry no
 *  precomposed `й` or `ё`, only the base letter and a combining mark). */
S3Text prepare_s3_text(const std::string& input, const UnicodeIndexer& indexer,
                       const S3TextConfig& cfg, Accentuator* accent);

}  // namespace nanotts
