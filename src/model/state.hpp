#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "model/bundle.hpp"
#include "numa/affinity.hpp"
#include "onnxruntime_cxx_api.h"

namespace xvibe {

// Recurrent state for one graph, held as raw bytes the runtime owns.
//
// Two things here matter for speed. The buffers are ping-ponged, so a step
// binds buffer A as input and B as output and then swaps, instead of copying
// the cache back and forth. And because K and V were split into separate
// tensors at export time, the batch axis is outermost in every state tensor,
// which makes one slot's slice contiguous -- admitting or evicting a stream
// from a batch is a single memcpy per tensor rather than a strided gather.
class StateBuffers {
 public:
  StateBuffers(const std::vector<StateSpec>& specs, int64_t batch, NumaAllocator* alloc);

  int64_t batch() const { return batch_; }
  size_t count() const { return specs_.size(); }
  const StateSpec& spec(size_t i) const { return specs_[i]; }

  void reset();                 // zero every slot
  void reset_slot(int64_t slot);

  Ort::Value input(size_t i, const Ort::MemoryInfo& mem);
  Ort::Value output(size_t i, const Ort::MemoryInfo& mem);
  void swap() { cur_ ^= 1; }

  // Raw access to one slot of one tensor in the current input buffer.
  uint8_t* slot_ptr(size_t i, int64_t slot);
  const uint8_t* slot_ptr(size_t i, int64_t slot) const;
  size_t slot_bytes(size_t i) const { return slot_bytes_[i]; }

  // Copy a whole single-slot state (a warmed voice, say) into one batch slot.
  void load_slot(int64_t slot, const StateBuffers& src, int64_t src_slot = 0);

 private:
  std::vector<StateSpec> specs_;
  int64_t batch_;
  int cur_ = 0;
  std::vector<std::vector<int64_t>> shapes_;  // per tensor, batch substituted
  std::vector<size_t> slot_bytes_;
  std::vector<size_t> total_bytes_;
  std::vector<NumaBuffer> storage_[2];
};

ONNXTensorElementDataType onnx_type_of(const std::string& dtype);

}  // namespace xvibe
