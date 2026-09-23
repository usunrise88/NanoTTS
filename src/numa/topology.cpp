#include "numa/topology.hpp"
#include <thread>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "numa/affinity.hpp"

namespace nanotts {
namespace fs = std::filesystem;

namespace {
std::string read_file(const fs::path& p) {
  std::ifstream f(p);
  if (!f) return {};
  std::string s;
  std::getline(f, s);
  return s;
}
}  // namespace

std::vector<int> parse_cpulist(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, ',')) {
    const auto dash = part.find('-');
    try {
      if (dash == std::string::npos) {
        out.push_back(std::stoi(part));
      } else {
        const int a = std::stoi(part.substr(0, dash));
        const int b = std::stoi(part.substr(dash + 1));
        for (int c = a; c <= b; ++c) out.push_back(c);
      }
    } catch (...) {
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

Topology Topology::detect() {
  Topology t;
  t.numa_available = numa_available_here();

  const fs::path node_root = "/sys/devices/system/node";
  if (fs::exists(node_root)) {
    for (const auto& e : fs::directory_iterator(node_root)) {
      const std::string name = e.path().filename().string();
      if (name.rfind("node", 0) != 0 || name.size() <= 4) continue;
      if (!std::all_of(name.begin() + 4, name.end(), ::isdigit)) continue;
      NumaNode n;
      n.id = std::stoi(name.substr(4));
      n.cpus = parse_cpulist(read_file(e.path() / "cpulist"));
      std::ifstream mem(e.path() / "meminfo");
      std::string line;
      while (std::getline(mem, line)) {
        const auto pos = line.find("MemTotal:");
        if (pos != std::string::npos) {
          std::istringstream is(line.substr(pos + 9));
          long long kb = 0;
          is >> kb;
          n.memory_mb = kb / 1024;
          break;
        }
      }
      if (!n.cpus.empty()) t.nodes.push_back(std::move(n));
    }
  }
  if (t.nodes.empty()) {
    NumaNode n;
    n.id = -1;
    const unsigned hw = std::thread::hardware_concurrency();
    for (unsigned i = 0; i < (hw ? hw : 1u); ++i) n.cpus.push_back(static_cast<int>(i));
    t.nodes.push_back(std::move(n));
  }
  std::sort(t.nodes.begin(), t.nodes.end(), [](auto& a, auto& b) { return a.id < b.id; });

  // L3 sharing, and one representative CPU per physical core.
  std::set<std::vector<int>> seen_l3;
  for (auto& node : t.nodes) {
    std::set<std::vector<int>> seen_core;
    for (int cpu : node.cpus) {
      const fs::path base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
      const auto l3 = parse_cpulist(read_file(base / "cache/index3/shared_cpu_list"));
      if (!l3.empty() && seen_l3.insert(l3).second) t.l3_groups.push_back(l3);
      auto siblings = parse_cpulist(read_file(base / "topology/thread_siblings_list"));
      if (siblings.empty()) siblings = {cpu};
      if (seen_core.insert(siblings).second) node.physical.push_back(siblings.front());
    }
    if (node.physical.empty()) node.physical = node.cpus;
  }
  return t;
}

std::string Topology::describe() const {
  std::ostringstream os;
  os << "NUMA " << (numa_available ? "available" : "unavailable") << ", " << nodes.size()
     << " node(s), " << l3_groups.size() << " L3 group(s)\n";
  for (const auto& n : nodes) {
    os << "  node " << n.id << ": " << n.cpus.size() << " cpus (" << n.physical.size()
       << " cores), " << n.memory_mb << " MB  [";
    for (size_t i = 0; i < n.cpus.size(); ++i) os << (i ? "," : "") << n.cpus[i];
    os << "]\n";
  }
  return os.str();
}

}  // namespace nanotts
