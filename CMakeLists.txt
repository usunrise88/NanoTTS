cmake_minimum_required(VERSION 3.22)
project(nanotts CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -fno-omit-frame-pointer")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Every third-party version is pinned on purpose: PocketTTS.cpp tracks dr_libs
# on `master`, which makes its builds unreproducible.
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(json)

FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(httplib)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(sentencepiece)

# ONNX Runtime comes from the prebuilt tarball unpacked in the build image.
set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime not found at '${ORT_ROOT}'. Set ONNXRUNTIME_ROOT.")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(LAME IMPORTED_TARGET lame)
find_library(NUMA_LIB numa)
find_path(NUMA_INCLUDE numa.h)

# The console is a Vite bundle compiled into the binary, so a deployment is one
# file. Turn it off to serve a directory with --web instead (handy in dev).
option(NANOTTS_EMBED_WEB "Compile the web console into the binary" ON)
set(NANOTTS_WEB_DIST "${CMAKE_CURRENT_SOURCE_DIR}/web/dist" CACHE PATH "Built web bundle")

if(NANOTTS_EMBED_WEB)
  if(NOT EXISTS "${NANOTTS_WEB_DIST}/index.html")
    message(FATAL_ERROR
      "NANOTTS_EMBED_WEB is on but ${NANOTTS_WEB_DIST}/index.html is missing.\n"
      "Build the console first:  cd web && npm ci && npm run build\n"
      "or configure with -DNANOTTS_EMBED_WEB=OFF to serve it from disk.")
  endif()
  file(GLOB_RECURSE NANOTTS_WEB_FILES "${NANOTTS_WEB_DIST}/*")
  set(NANOTTS_WEB_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/web_assets_embedded.cpp")
  add_custom_command(
    OUTPUT "${NANOTTS_WEB_GEN}"
    COMMAND ${CMAKE_COMMAND} -DINPUT_DIR=${NANOTTS_WEB_DIST} -DOUTPUT=${NANOTTS_WEB_GEN}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_web.cmake"
    DEPENDS ${NANOTTS_WEB_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_web.cmake"
    COMMENT "Embedding web console"
    VERBATIM)
  set(NANOTTS_WEB_SOURCE "${NANOTTS_WEB_GEN}")
  message(STATUS "web console: embedded from ${NANOTTS_WEB_DIST}")
else()
  set(NANOTTS_WEB_SOURCE src/http/web_assets_none.cpp)
  message(STATUS "web console: not embedded (serve with --web)")
endif()

add_library(nanotts_core STATIC
  src/model/bundle.cpp
  src/model/state.cpp
  src/model/tokenizer.cpp
  src/engine/engine.cpp
  src/engine/s3.cpp
  src/engine/factory.cpp
  src/text/accent.cpp
  src/text/text.cpp
  src/text/numbers.cpp
  src/text/s3_text.cpp
  src/voice/store.cpp
  src/voice/safetensors.cpp
  src/audio/wav.cpp
  src/audio/analyse.cpp
  src/audio/resample.cpp
  src/audio/mp3.cpp
  src/numa/topology.cpp
  src/numa/affinity.cpp
  ${NANOTTS_WEB_SOURCE}
)
target_include_directories(nanotts_core PUBLIC
  src
  "${ORT_ROOT}/include"
  "${sentencepiece_SOURCE_DIR}/src")
target_link_libraries(nanotts_core PUBLIC
  nlohmann_json::nlohmann_json
  httplib::httplib
  sentencepiece-static
  "${ORT_ROOT}/lib/libonnxruntime.so"
  pthread dl)

if(NUMA_LIB)
  target_compile_definitions(nanotts_core PUBLIC NANOTTS_HAVE_NUMA=1)
  target_include_directories(nanotts_core PUBLIC "${NUMA_INCLUDE}")
  target_link_libraries(nanotts_core PUBLIC "${NUMA_LIB}")
  message(STATUS "libnuma: ${NUMA_LIB}")
else()
  message(WARNING "libnuma not found - NUMA pinning will be a no-op")
endif()

if(LAME_FOUND)
  target_compile_definitions(nanotts_core PUBLIC NANOTTS_HAVE_LAME=1)
  target_link_libraries(nanotts_core PUBLIC PkgConfig::LAME)
  message(STATUS "libmp3lame: found")
else()
  message(WARNING "libmp3lame not found - mp3 output disabled")
endif()

add_executable(nanotts src/main.cpp src/http/server.cpp)
target_link_libraries(nanotts PRIVATE nanotts_core httplib::httplib)
set_target_properties(nanotts PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS nanotts RUNTIME DESTINATION bin)
