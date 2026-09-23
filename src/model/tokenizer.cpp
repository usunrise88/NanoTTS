#include "model/tokenizer.hpp"

#include <sentencepiece_processor.h>

#include <stdexcept>

namespace nanotts {

Tokenizer::Tokenizer(const std::filesystem::path& model_path)
    : sp_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
  const auto status = sp_->Load(model_path.string());
  if (!status.ok()) throw std::runtime_error("tokenizer: " + status.ToString());
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  const auto status = sp_->Encode(text, &ids);
  if (!status.ok()) throw std::runtime_error("encode: " + status.ToString());
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  const auto status = sp_->Decode(ids, &out);
  if (!status.ok()) throw std::runtime_error("decode: " + status.ToString());
  return out;
}

int Tokenizer::vocab_size() const { return sp_->GetPieceSize(); }

}  // namespace nanotts
