#pragma once
#include <string>
#include <vector>

namespace nanotts {

struct NumaNode {
  int id = 0;
  std::vector<int> cpus;        // logical CPUs, SMT siblings included
  std::vector<int> physical;    // one CPU per physical core
  long long memory_mb = 0;
};

struct Topology {
  std::vector<NumaNode> nodes;
  std::vector<std::vector<int>> l3_groups;  // CPUs sharing one L3 (a CCX on AMD)
  bool numa_available = false;

  static Topology detect();
  std::string describe() const;
};

// Parse a Linux cpulist such as "0-3,16-19".
std::vector<int> parse_cpulist(const std::string& s);

}  // namespace nanotts
