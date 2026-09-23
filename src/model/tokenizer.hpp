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

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
};

}  // namespace nanotts
