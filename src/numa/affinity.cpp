#include "numa/affinity.hpp"

#include <sched.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>
#include <new>

#if NANOTTS_HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace nanotts {

namespace {
constexpr size_t kHugePageThreshold = 2u << 20;  // ask for THP above 2 MiB
}

NumaBuffer::NumaBuffer(size_t bytes, int node) : size_(bytes) {
  if (bytes == 0) return;
#if NANOTTS_HAVE_NUMA
  if (node >= 0 && numa_available() != -1) {
    data_ = static_cast<uint8_t*>(numa_alloc_onnode(bytes, node));
    numa_owned_ = data_ != nullptr;
  }
#endif
  if (!data_) {
    if (posix_memalign(reinterpret_cast<void**>(&data_), 64, bytes) != 0) throw std::bad_alloc();
    numa_owned_ = false;
  }
  if (bytes >= kHugePageThreshold) {
    // Best-effort: THP is in `madvise` mode on the target host.
    ::madvise(data_, bytes, MADV_HUGEPAGE);
  }
  std::memset(data_, 0, bytes);
}

void NumaBuffer::release() {
  if (!data_) return;
#if NANOTTS_HAVE_NUMA
  if (numa_owned_) {
    numa_free(data_, size_);
    data_ = nullptr;
    size_ = 0;
    return;
  }
#endif
  std::free(data_);
  data_ = nullptr;
  size_ = 0;
}

NumaBuffer::~NumaBuffer() { release(); }

NumaBuffer::NumaBuffer(NumaBuffer&& o) noexcept
    : data_(o.data_), size_(o.size_), numa_owned_(o.numa_owned_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

NumaBuffer& NumaBuffer::operator=(NumaBuffer&& o) noexcept {
  if (this != &o) {
    release();
    data_ = o.data_;
    size_ = o.size_;
    numa_owned_ = o.numa_owned_;
    o.data_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

void NumaBuffer::zero() {
  if (data_) std::memset(data_, 0, size_);
}

bool pin_thread(const std::vector<int>& cpus) {
  if (cpus.empty()) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : cpus) CPU_SET(c, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}

bool bind_memory_to_node(int node) {
#if NANOTTS_HAVE_NUMA
  if (node < 0 || numa_available() == -1) return false;
  // First-touch allocation follows the calling thread's policy, so setting it
  // before the ORT session is created is what puts the weights on this node.
  unsigned long mask = 1ul << node;
  return set_mempolicy(MPOL_BIND, &mask, sizeof(mask) * 8) == 0;
#else
  (void)node;
  return false;
#endif
}

void unbind_memory() {
#if NANOTTS_HAVE_NUMA
  set_mempolicy(MPOL_DEFAULT, nullptr, 0);
#endif
}

bool numa_available_here() {
#if NANOTTS_HAVE_NUMA
  return numa_available() != -1;
#else
  return false;
#endif
}

}  // namespace nanotts
