#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nanotts {

// Memory that is bound to one NUMA node. On a 4-node single-socket EPYC a
// remote read costs roughly 60% more than a local one, so every buffer a
// worker touches on the hot path is allocated on that worker's node and huge
// pages are requested for the large ones.
class NumaBuffer {
 public:
  NumaBuffer() = default;
  NumaBuffer(size_t bytes, int node);
  ~NumaBuffer();
  NumaBuffer(NumaBuffer&&) noexcept;
  NumaBuffer& operator=(NumaBuffer&&) noexcept;
  NumaBuffer(const NumaBuffer&) = delete;
  NumaBuffer& operator=(const NumaBuffer&) = delete;

  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  void zero();

 private:
  void release();
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
  bool numa_owned_ = false;
};

class NumaAllocator {
 public:
  explicit NumaAllocator(int node) : node_(node) {}
  NumaBuffer allocate(size_t bytes) const { return NumaBuffer(bytes, node_); }
  int node() const { return node_; }

 private:
  int node_;
};

// Pin the calling thread to `cpus` and make its allocations land on `node`.
// Returns false when the capability is missing (containers without
// CAP_SYS_NICE, kernels without libnuma) rather than aborting: the service
// still works, it just loses locality.
bool pin_thread(const std::vector<int>& cpus);
bool bind_memory_to_node(int node);
void unbind_memory();

bool numa_available_here();

}  // namespace nanotts
