#pragma once
#include <span>
#include <string_view>

namespace nanotts {

// One file from the built console, compiled into the binary.
struct WebAsset {
  std::string_view path;  // relative to the bundle root, e.g. "assets/index.js"
  std::string_view mime;
  std::string_view body;
};

// Empty when the build was configured with -DNANOTTS_EMBED_WEB=OFF, in which
// case the server falls back to serving a directory given by --web.
std::span<const WebAsset> embedded_web();

}  // namespace nanotts
