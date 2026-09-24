# Runtime image. Three stages: the console bundle, the native binary that
# embeds it, and a runtime layer carrying only the executable, ONNX Runtime and
# the two codec libraries.
FROM node:22-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates curl pkg-config \
        libssl-dev \
        libnuma-dev libmp3lame-dev \
    && rm -rf /var/lib/apt/lists/*

ARG ORT_VERSION=1.30.0
RUN curl -sL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY src ./src
COPY --from=web /web/dist ./web/dist
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && strip build/nanotts

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libnuma1 libmp3lame0 numactl ca-certificates curl libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 -m -d /var/lib/nanotts nanotts

COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /src/build/nanotts /usr/local/bin/nanotts
RUN ldconfig

USER nanotts
WORKDIR /var/lib/nanotts
EXPOSE 8080

# curl, not a /dev/tcp redirect: /bin/sh here is dash, which has no /dev/tcp,
# so the redirect form reported every container unhealthy for its whole life.
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s \
  CMD ["curl", "-fsS", "-o", "/dev/null", "http://127.0.0.1:8080/healthz"]

ENTRYPOINT ["/usr/local/bin/nanotts"]
# No --web: the console is compiled into the binary.
CMD ["serve", "--bundle", "/models/bundle", "--tokenizer", "/models/bundle/tokenizer.model", \
     "--voices", "/data/voices", "--port", "8080"]
