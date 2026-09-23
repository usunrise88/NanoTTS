#include "model/state.hpp"

#include <cstring>
#include <stdexcept>

namespace nanotts {

ONNXTensorElementDataType onnx_type_of(const std::string& dtype) {
  if (dtype == "float32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (dtype == "float16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (dtype == "int64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  if (dtype == "bool") return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  throw std::runtime_error("unsupported state dtype: " + dtype);
}

StateBuffers::StateBuffers(const std::vector<StateSpec>& specs, int64_t batch,
                           NumaAllocator* alloc)
    : specs_(specs), batch_(batch) {
  shapes_.reserve(specs_.size());
  for (const auto& s : specs_) {
    std::vector<int64_t> shape = s.shape;
    if (shape.empty()) shape.push_back(batch);
    shape[0] = batch;
    shapes_.push_back(shape);

    int64_t per_slot = 1;
    for (size_t i = 1; i < shape.size(); ++i) per_slot *= shape[i];
    const size_t slot_bytes = static_cast<size_t>(per_slot) * s.element_size();
    slot_bytes_.push_back(slot_bytes);
    total_bytes_.push_back(slot_bytes * static_cast<size_t>(batch));
  }

  for (int b = 0; b < 2; ++b) {
    storage_[b].reserve(specs_.size());
    for (size_t i = 0; i < specs_.size(); ++i)
      storage_[b].push_back(alloc->allocate(total_bytes_[i]));
  }
}

void StateBuffers::reset() {
  for (int b = 0; b < 2; ++b)
    for (auto& buf : storage_[b]) buf.zero();
  cur_ = 0;
}

void StateBuffers::reset_slot(int64_t slot) {
  for (int b = 0; b < 2; ++b)
    for (size_t i = 0; i < specs_.size(); ++i)
      std::memset(storage_[b][i].data() + slot_bytes_[i] * static_cast<size_t>(slot), 0,
                  slot_bytes_[i]);
}

Ort::Value StateBuffers::input(size_t i, const Ort::MemoryInfo& mem) {
  return Ort::Value::CreateTensor(mem, storage_[cur_][i].data(), total_bytes_[i],
                                  shapes_[i].data(), shapes_[i].size(),
                                  onnx_type_of(specs_[i].dtype));
}

Ort::Value StateBuffers::output(size_t i, const Ort::MemoryInfo& mem) {
  return Ort::Value::CreateTensor(mem, storage_[cur_ ^ 1][i].data(), total_bytes_[i],
                                  shapes_[i].data(), shapes_[i].size(),
                                  onnx_type_of(specs_[i].dtype));
}

uint8_t* StateBuffers::slot_ptr(size_t i, int64_t slot) {
  return storage_[cur_][i].data() + slot_bytes_[i] * static_cast<size_t>(slot);
}

const uint8_t* StateBuffers::slot_ptr(size_t i, int64_t slot) const {
  return storage_[cur_][i].data() + slot_bytes_[i] * static_cast<size_t>(slot);
}

void StateBuffers::load_slot(int64_t slot, const StateBuffers& src, int64_t src_slot) {
  if (src.specs_.size() != specs_.size()) throw std::runtime_error("state layout mismatch");
  for (size_t i = 0; i < specs_.size(); ++i) {
    if (src.slot_bytes_[i] != slot_bytes_[i]) throw std::runtime_error("state slot size mismatch");
    std::memcpy(slot_ptr(i, slot), src.slot_ptr(i, src_slot), slot_bytes_[i]);
  }
}

}  // namespace nanotts
