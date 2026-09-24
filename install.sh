#!/usr/bin/env bash
# NanoTTS installer.
#
#   curl -fsSL https://raw.githubusercontent.com/usunrise88/NanoTTS/main/install.sh | bash
#
# Installs a CPU speech-synthesis server for Kyutai Pocket TTS and its
# fine-tunes: fetches the source, exports the checkpoint to ONNX, builds the
# runtime image and starts it. Everything runs in Docker, so the only thing this
# script puts on the host is a directory and a launcher.
#
#   ... | bash -s -- --model english        a Pocket TTS base language
#   ... | bash -s -- update                 rebuild at the latest revision
#   ... | bash -s -- uninstall              remove everything but the voices
#   ... | bash -s -- uninstall --purge      remove the voices too
set -euo pipefail

REPO_URL="${NANOTTS_REPO:-https://github.com/usunrise88/NanoTTS.git}"
REF="${NANOTTS_REF:-main}"
MODEL="${NANOTTS_MODEL:-genvoice/xVibePocketTTS}"
PORT="${NANOTTS_PORT:-8080}"
BIND="${NANOTTS_BIND:-127.0.0.1}"
PRECISION="${NANOTTS_PRECISION:-int8}"
ACCENT="${NANOTTS_ACCENT:-auto}"
PREFIX="${NANOTTS_PREFIX:-}"
ACTION=""
PURGE=0
ASSUME_YES=0

# ---------------------------------------------------------------- output

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; R=$'\033[0m'
else
  B=""; DIM=""; RED=""; GRN=""; YLW=""; R=""
fi
step() { printf '%s==>%s %s\n' "$B" "$R" "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$YLW" "$R" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }
ok()   { printf '%s  ok%s %s\n' "$GRN" "$R" "$*"; }

usage() {
  cat <<EOF
${B}NanoTTS installer${R}

  install.sh [install|update|uninstall] [options]

${B}Options${R}
  --model NAME        Pocket TTS language (english, german, spanish_24l, ...),
                      a HuggingFace repo (owner/name), or an hf:// / https:// /
                      local path to a config.yaml.
                      default: ${MODEL}
  --port PORT         default: ${PORT}
  --bind ADDR         address to publish the port on; 0.0.0.0 exposes it to the
                      network. default: ${BIND}
  --prefix DIR        install location. default: /opt/nanotts as root,
                      ~/.nanotts otherwise
  --precision int8|fp32
                      int8 is ~30% faster; see the README on what it costs.
                      default: ${PRECISION}
  --accent auto|on|off
                      RUAccent stress sidecar. Russian checkpoints need it --
                      without stress marks the same model scores 17.7% WER
                      instead of 3.2%. Useless for other languages.
                      default: ${ACCENT}
  --ref REF           git ref to install. default: ${REF}
  --purge             uninstall: delete warmed voices as well
  --yes               do not ask for confirmation
  --help

Every option also reads from an environment variable: NANOTTS_MODEL,
NANOTTS_PORT, NANOTTS_BIND, NANOTTS_PREFIX, NANOTTS_PRECISION, NANOTTS_ACCENT,
NANOTTS_REF.
EOF
}

# ---------------------------------------------------------------- arguments

# An update reuses the choices made at install time, so it has to know which of
# the settings below the caller actually asked for this time round.
EXPLICIT=""
mark() { EXPLICIT="$EXPLICIT $1"; }
given() { case " $EXPLICIT " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    install|update|uninstall) ACTION="$1" ;;
    --model)     MODEL="${2:?--model needs a value}"; mark MODEL; shift ;;
    --port)      PORT="${2:?--port needs a value}"; mark PORT; shift ;;
    --bind)      BIND="${2:?--bind needs a value}"; mark BIND; shift ;;
    --prefix)    PREFIX="${2:?--prefix needs a value}"; shift ;;
    --precision) PRECISION="${2:?--precision needs a value}"; mark PRECISION; shift ;;
    --accent)    ACCENT="${2:?--accent needs a value}"; mark ACCENT; shift ;;
    --ref)       REF="${2:?--ref needs a value}"; mark REF; shift ;;
    --purge)     PURGE=1 ;;
    --yes|-y)    ASSUME_YES=1 ;;
    --help|-h)   usage; exit 0 ;;
    *)           die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done
ACTION="${ACTION:-install}"
for v in MODEL PORT BIND PRECISION ACCENT REF; do
  eval "[ -n \"\${NANOTTS_$v:-}\" ]" && mark "$v"
done
true

case "$PRECISION" in int8|fp32) ;; *) die "--precision must be int8 or fp32" ;; esac
case "$ACCENT" in auto|on|off) ;; *) die "--accent must be auto, on or off" ;; esac
case "$PORT" in ''|*[!0-9]*) die "--port must be a number" ;; esac

if [ -z "$PREFIX" ]; then
  if [ "$(id -u)" = 0 ]; then PREFIX=/opt/nanotts; else PREFIX="$HOME/.nanotts"; fi
fi
SRC="$PREFIX/src"
BUNDLE="$PREFIX/bundle"
DATA="$PREFIX/voices"
COMPOSE="$PREFIX/compose.yml"
STATE="$PREFIX/install.env"

# `docker compose` on anything current, `docker-compose` on older hosts.
compose() {
  if docker compose version >/dev/null 2>&1; then docker compose "$@"
  else docker-compose "$@"; fi
}

confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  [ -t 0 ] || die "$1 needs confirmation; rerun with --yes or from a terminal"
  printf '%s [y/N] ' "$1"; read -r reply
  case "$reply" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------- preflight

preflight() {
  step "Checking the host"

  [ "$(uname -s)" = Linux ] || die "NanoTTS serves on Linux only (found $(uname -s))"
  case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die "x86_64 only; ONNX Runtime's int8 path here is MLAS/AVX2 (found $(uname -m))" ;;
  esac

  # The AR step is a stream of GEMVs through MLAS. Without AVX2 this runs, but
  # slowly enough that it is worth saying so up front.
  if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
    ok "AVX2"
  else
    warn "no AVX2: synthesis will be several times slower than the published numbers"
  fi

  command -v docker >/dev/null 2>&1 || die "docker is required: https://docs.docker.com/engine/install/"
  docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (is it running, are you in the docker group?)"
  ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo present)"

  compose version >/dev/null 2>&1 || die "docker compose v2 (or docker-compose) is required"
  command -v git >/dev/null 2>&1 || die "git is required"

  local avail
  avail=$(df -Pk "$(dirname "$PREFIX")" 2>/dev/null | awk 'NR==2{print int($4/1048576)}') || avail=0
  if [ "${avail:-0}" -lt 12 ]; then
    warn "only ${avail} GiB free under $(dirname "$PREFIX"); the export needs about 12 GiB"
  else
    ok "${avail} GiB free"
  fi

  local memgb
  memgb=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 0)
  [ "${memgb:-0}" -ge 4 ] || warn "only ${memgb} GiB of RAM; the ONNX export peaks around 6 GiB"
}

# ---------------------------------------------------------------- model

# Checkpoints that ship ONNX need fetching, not tracing: Supertonic 3 and
# TeraTTS v2 release graphs rather than weights, so they take a different path
# through the export step and a different backend at runtime.
is_s3_model() {
  case "$MODEL" in
    Supertone/*|*upertonic*|TeraSpace/*|*eraTTS*) return 0 ;;
    *) return 1 ;;
  esac
}

# Turns whatever the user passed into the export's arguments.
#   english, spanish_24l     -> --language NAME   (built into pocket-tts)
#   owner/name               -> --config hf://owner/name/config.yaml
#   hf://..., https://..., / -> --config AS GIVEN
model_args() {
  case "$MODEL" in
    hf://*|https://*|/*|./*) printf -- '--config %s' "$MODEL" ;;
    */*)                     printf -- '--config hf://%s/config.yaml' "$MODEL" ;;
    *)                       printf -- '--language %s' "$MODEL" ;;
  esac
}

# Russian is the only checkpoint whose text needs stress marks, and RUAccent is
# a Russian model, so anything else gets the sidecar switched off.
wants_accent() {
  case "$ACCENT" in
    on)  return 0 ;;
    off) return 1 ;;
  esac
  case "$MODEL" in
    # TeraTTS v2 is Russian and English and was trained on stressed text;
    # Supertonic is en/ko/ja and has no use for a Russian accentuator.
    *[Rr]ussian*|*ussian*|*xVibe*|*XVibe*|genvoice/*|TeraSpace/*|*eraTTS*) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------- install

fetch_source() {
  step "Fetching the source at ${REF}"
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" remote set-url origin "$REPO_URL"
    git -C "$SRC" fetch --depth 1 origin "$REF"
    git -C "$SRC" checkout -q FETCH_HEAD
  else
    rm -rf "$SRC"
    git clone --depth 1 --branch "$REF" "$REPO_URL" "$SRC" 2>/dev/null \
      || git clone --depth 1 "$REPO_URL" "$SRC"
  fi
  ok "$(git -C "$SRC" rev-parse --short HEAD)"
}

build_bundle() {
  if [ -f "$BUNDLE/bundle.json" ] && [ "${REBUILD_BUNDLE:-0}" != 1 ]; then
    info "bundle already present, skipping the export (delete $BUNDLE to redo it)"
    return
  fi
  mkdir -p "$BUNDLE"
  docker build -q -t nanotts-export "$SRC/tools/export" >/dev/null

  if is_s3_model; then
    step "Fetching ${MODEL}"
    info "this release ships ONNX, so there is nothing to trace -- only ~640 MB to download"
    mkdir -p "$DATA"
    docker run --rm -v "$SRC/tools/export:/work" -v "$BUNDLE:/out" -v "$DATA:/voices" -w /work \
      nanotts-export python fetch_s3.py --model "$MODEL" --out /out --voices /voices
    [ -f "$BUNDLE/bundle.json" ] || die "the fetch produced no bundle.json"
    ok "$(du -sh "$BUNDLE" | cut -f1) in $BUNDLE"
    return
  fi

  step "Exporting ${MODEL} to ONNX"
  info "this downloads the checkpoint and traces five graphs; it takes a few minutes"

  # beartype rejects the symbolic tensors the tracer feeds through, and the
  # export is the one place that matters.
  # shellcheck disable=SC2046
  docker run --rm -e POCKET_TTS_NO_BEARTYPE=1 \
    -v "$SRC/tools/export:/work" -v "$BUNDLE:/out" -w /work \
    nanotts-export python export_onnx.py $(model_args)

  if [ "$PRECISION" = int8 ]; then
    step "Quantising the two graphs that are read every frame"
    docker run --rm -v "$SRC/tools/export:/work" -v "$BUNDLE:/out" -w /work \
      nanotts-export python quantize.py
  fi

  [ -f "$BUNDLE/bundle.json" ] || die "the export produced no bundle.json"
  # The export writes the tokenizer next to the graphs, which is where the
  # runtime looks for it.
  [ -f "$BUNDLE/tokenizer.model" ] || die "the export produced no tokenizer.model"
  ok "$(du -sh "$BUNDLE" | cut -f1) in $BUNDLE"
}

# A bundle carries no voices, and a server with none cannot answer a request.
# Checkpoints that ship their own get those; anything else gets the voice
# pocket-tts would pick for that language.
build_voices() {
  if is_s3_model; then
    info "voices came with the checkpoint"
    return
  fi
  if [ -n "$(ls -A "$DATA" 2>/dev/null)" ]; then
    info "voices already present, leaving them alone"
    return
  fi
  step "Warming a voice to start with"
  mkdir -p "$DATA"

  local sources=""
  case "$MODEL" in
    */*)
      if [ "${MODEL#hf://}" = "$MODEL" ] && [ "${MODEL#/}" = "$MODEL" ] \
         && [ "${MODEL#https://}" = "$MODEL" ]; then
        sources=$(curl -fsS "https://huggingface.co/api/models/${MODEL}" 2>/dev/null \
          | tr ',' '\n' | grep -o '"rfilename":"voices/[^"]*\.safetensors"' \
          | sed 's/.*:"/hf:\/\/'"${MODEL//\//\\/}"'\//; s/"$//' | tr '\n' ' ') || sources=""
      fi
      ;;
  esac

  # shellcheck disable=SC2046,SC2086
  docker run --rm -e POCKET_TTS_NO_BEARTYPE=1 \
    -v "$SRC/tools/export:/work" -v "$BUNDLE:/out:ro" -v "$DATA:/voices" -w /work \
    nanotts-export python make_voices.py $(model_args) $sources \
    || warn "no starting voice could be made; upload one through POST /v1/voices"
}

write_compose() {
  step "Writing $COMPOSE"
  mkdir -p "$DATA" "$PREFIX/models"

  local accent_service="" accent_flag="" depends=""
  if wants_accent; then
    accent_flag="      - --accent-url=http://accent:8100"
    depends="    depends_on: [accent]"
    accent_service="  accent:
    build:
      context: ./src/accent
    image: nanotts-accent:latest
    environment:
      RUACCENT_POOL: \"3\"
      RUACCENT_THREADS: \"4\"
    restart: unless-stopped
    networks: [nanotts]
"
  fi

  local precision_flag=""
  [ "$PRECISION" = int8 ] && precision_flag="      - --int8"

  # The two architectures name their text front-end differently: a sentencepiece
  # model for pocket, a codepoint table for s3.
  local tokenizer_file="/models/bundle/tokenizer.model"
  [ -f "$BUNDLE/unicode_indexer.json" ] && tokenizer_file="/models/bundle/unicode_indexer.json"

  # The voice directory is a bind mount, so the container has to write it as
  # whoever owns it on the host. Left as the image's own user, uploads through
  # POST /v1/voices fail and warmed voices come back unreadable -- which looks
  # exactly like having no voices at all.
  local run_as
  run_as="    user: \"$(id -u):$(id -g)\""

  cat > "$COMPOSE" <<EOF
# Generated by install.sh, and regenerated on update -- edits here are lost.
# Change settings by rerunning install.sh with the flag you want.
services:
${accent_service}  nanotts:
    build:
      context: ./src
      dockerfile: docker/Dockerfile
    image: nanotts:latest
${depends}
${run_as}
    ports: ["${BIND}:${PORT}:8080"]
    volumes:
      - ./bundle:/models/bundle:ro
      - ./voices:/data/voices
      # Extra checkpoints the console downloads and switches to at runtime.
      - ./models:/models/registry
    # NUMA pinning needs CAP_SYS_NICE for sched_setaffinity. Without it the
    # service still runs, it just loses locality and says so at startup.
    cap_add: [SYS_NICE]
    command:
      - serve
      - --bundle=/models/bundle
      - --tokenizer=${tokenizer_file}
      - --voices=/data/voices
      - --models=/models/registry
      - --port=8080
${precision_flag}
${accent_flag}
    restart: unless-stopped
    ulimits:
      memlock: -1
    networks: [nanotts]

networks:
  nanotts:
EOF
  # Blank lines from the unset optional flags above.
  sed -i '/^$/d' "$COMPOSE"
  ok "written"
}

install_launcher() {
  local bindir target
  if [ "$(id -u)" = 0 ]; then bindir=/usr/local/bin; else bindir="$HOME/.local/bin"; fi
  mkdir -p "$bindir"
  target="$bindir/nanotts"

  cat > "$target" <<EOF
#!/usr/bin/env bash
# NanoTTS launcher, written by install.sh.
set -euo pipefail
PREFIX="$PREFIX"
compose() {
  if docker compose version >/dev/null 2>&1; then docker compose -f "\$PREFIX/compose.yml" "\$@"
  else docker-compose -f "\$PREFIX/compose.yml" "\$@"; fi
}
case "\${1:-status}" in
  start)   compose up -d ;;
  stop)    compose down ;;
  restart) compose restart ;;
  logs)    shift; compose logs -f "\$@" ;;
  status)  compose ps ;;
  update)    exec bash "\$PREFIX/install.sh" update ;;
  uninstall) shift; exec bash "\$PREFIX/install.sh" uninstall "\$@" ;;
  *) echo "usage: nanotts {start|stop|restart|status|logs|update|uninstall}" >&2; exit 2 ;;
esac
EOF
  chmod +x "$target"
  cp "$SRC/install.sh" "$PREFIX/install.sh" 2>/dev/null || true
  ok "$target"

  case ":$PATH:" in
    *":$bindir:"*) ;;
    *) warn "$bindir is not on your PATH; add it or call $target directly" ;;
  esac
}

save_state() {
  cat > "$STATE" <<EOF
NANOTTS_REPO=$REPO_URL
NANOTTS_MODEL=$MODEL
NANOTTS_PORT=$PORT
NANOTTS_BIND=$BIND
NANOTTS_PRECISION=$PRECISION
NANOTTS_ACCENT=$ACCENT
NANOTTS_REF=$REF
EOF
}

start_and_wait() {
  step "Building the image and starting"
  ( cd "$PREFIX" && compose -f "$COMPOSE" up -d --build )

  printf '    waiting for the server'
  local waited=0
  while [ "$waited" -lt 180 ]; do
    if curl -fsS "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
      printf '\n'; ok "ready"
      return 0
    fi
    printf '.'; sleep 2; waited=$((waited + 2))
  done
  printf '\n'
  warn "the server did not come up in 180s; check: nanotts logs"
  return 1
}

do_install() {
  preflight
  mkdir -p "$PREFIX"
  fetch_source
  build_bundle
  build_voices
  write_compose
  install_launcher
  save_state
  start_and_wait || true

  local url="http://${BIND}:${PORT}" accent_state
  if wants_accent; then
    accent_state="RUAccent sidecar (Russian)"
  else
    accent_state="off -- supply stress marks yourself if the checkpoint wants them"
  fi
  [ "$BIND" = 0.0.0.0 ] && url="http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}"
  cat <<EOF

${B}NanoTTS is installed.${R}

  console   ${url}
  API       ${url}/v1/audio/speech        (OpenAI-shaped)
  model     ${MODEL}
  stress    ${accent_state}
  files     ${PREFIX}

  ${DIM}nanotts status | logs | restart | update | uninstall${R}

Try it:

  curl -X POST ${url}/v1/audio/speech -H 'Content-Type: application/json' \\
    -d '{"input":"Hello from NanoTTS.","voice":"male_deep"}' --output out.wav
EOF
}

# ---------------------------------------------------------------- update

do_update() {
  [ -d "$SRC/.git" ] || die "nothing installed at $PREFIX (run install first)"
  # Reuse the choices made at install time unless this run overrode them. The
  # state file only sets NANOTTS_*, so the working variables have to be taken
  # from it explicitly -- sourcing it alone silently changed nothing.
  if [ -f "$STATE" ]; then
    # shellcheck disable=SC1090
    . "$STATE"
    given MODEL     || MODEL="${NANOTTS_MODEL:-$MODEL}"
    given PORT      || PORT="${NANOTTS_PORT:-$PORT}"
    given BIND      || BIND="${NANOTTS_BIND:-$BIND}"
    given PRECISION || PRECISION="${NANOTTS_PRECISION:-$PRECISION}"
    given ACCENT    || ACCENT="${NANOTTS_ACCENT:-$ACCENT}"
    given REF       || REF="${NANOTTS_REF:-$REF}"
    REPO_URL="${NANOTTS_REPO:-$REPO_URL}"
  fi
  preflight

  local before after
  before=$(git -C "$SRC" rev-parse HEAD)
  fetch_source
  after=$(git -C "$SRC" rev-parse HEAD)

  if [ "$before" = "$after" ]; then
    info "already at $(git -C "$SRC" rev-parse --short HEAD)"
  else
    info "$(git -C "$SRC" rev-parse --short "$before") -> $(git -C "$SRC" rev-parse --short "$after")"
  fi

  # The bundle is a product of the checkpoint, not of this repository, so an
  # update leaves it alone. Re-export explicitly if the graphs change.
  write_compose
  install_launcher
  save_state
  start_and_wait || true
  ok "updated"
  info "the ONNX bundle was left as it is; delete $BUNDLE and rerun install to rebuild it"
}

# ---------------------------------------------------------------- uninstall

do_uninstall() {
  [ -e "$PREFIX" ] || die "nothing at $PREFIX"
  # Refuse to touch anything that is not ours.
  case "$PREFIX" in
    /|/usr|/usr/*|/etc|/etc/*|/var|/var/*|"$HOME") die "refusing to remove $PREFIX" ;;
  esac

  echo "This removes:"
  echo "  - the NanoTTS containers and images"
  echo "  - $SRC, $BUNDLE and $COMPOSE"
  if [ "$PURGE" = 1 ]; then
    echo "  - ${RED}$DATA, including every warmed voice${R}"
  else
    echo "  - $DATA is kept (add --purge to delete it too)"
  fi
  confirm "Continue?" || { info "cancelled"; exit 0; }

  step "Stopping"
  [ -f "$COMPOSE" ] && ( cd "$PREFIX" && compose -f "$COMPOSE" down --remove-orphans ) || true

  step "Removing images"
  docker image rm -f nanotts:latest nanotts-accent:latest nanotts-export:latest >/dev/null 2>&1 || true

  step "Removing files"
  rm -rf "$SRC" "$BUNDLE" "$COMPOSE" "$STATE" "$PREFIX/install.sh"
  if [ "$PURGE" = 1 ]; then
    rm -rf "$DATA"
    rmdir "$PREFIX" 2>/dev/null || true
  fi

  for bindir in /usr/local/bin "$HOME/.local/bin"; do
    [ -f "$bindir/nanotts" ] && rm -f "$bindir/nanotts"
  done

  ok "removed"
  [ "$PURGE" = 1 ] || info "voices left in $DATA"
}

case "$ACTION" in
  install)   do_install ;;
  update)    do_update ;;
  uninstall) do_uninstall ;;
esac
