#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace nanotts {

// Google SentencePiece, used directly rather than reimplemented: the Russian
// model carries U+0301 as a dedicated piece (id 3) and the chunker depends on
// its exact segmentation of ".!...?" and ",;:".
class Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path& model_path);
  ~Tokenizer();

  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  int vocab_size() const;
  /** True when this piece of text encodes to the unknown token. The Russian
   *  checkpoint's vocabulary is narrower than real text: straight and
   *  typographic quotes, en dashes and brackets are all absent, and feeding
   *  their <unk> to the model is worse than not sending them at all. */
  bool is_unknown(const std::string& text) const;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
};

}  // namespace nanotts
