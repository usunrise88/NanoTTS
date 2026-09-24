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

bool Tokenizer::is_unknown(const std::string& text) const {
  // Encoded between two letters the model certainly knows: a character on its
  // own encodes differently, because SentencePiece's word-boundary marker gets
  // in the way. The only question asked is whether the unknown id appears --
  // counting tokens instead would call an ordinary letter unknown whenever it
  // happens to merge with its neighbours into one piece.
  static const std::string guard = "\xd0\xb0";  // а
  std::vector<int> ids;
  if (!sp_->Encode(guard + text + guard, &ids).ok()) return true;
  const int unk = sp_->unk_id();
  for (int id : ids)
    if (id == unk) return true;
  return false;
}

}  // namespace nanotts
