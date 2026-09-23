"""RUAccent as a sidecar.

Russian stress lives here rather than in the C++ runtime for one reason: the
checkpoint was trained on text carrying U+0301, and dropping it costs 17.7% WER
against 3.2% on long sentences. RUAccent is Python and has no native port, so it
gets its own process instead of being reimplemented badly.

Pinned to the revision GenVoice pin, because the dictionary and the omograph
model decide how the words come out.

  POST /accent  {"texts": ["..."], "skip_regex": "..."} -> {"texts": ["..."]}
  GET  /healthz
"""

import json
import logging
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from huggingface_hub import snapshot_download
import ruaccent as ruaccent_package
from ruaccent import RUAccent

REPO = os.environ.get("RUACCENT_REPO", "ruaccent/accentuator")
REVISION = os.environ.get("RUACCENT_REVISION", "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855")
MODEL_SIZE = os.environ.get("RUACCENT_MODEL_SIZE", "turbo3.1")
WORKDIR = Path(os.environ.get("RUACCENT_WORKDIR", "/var/lib/ruaccent"))
# onnxruntime releases the GIL, so a pool of instances gives real concurrency in
# one process. Each instance holds its own models, so this trades memory for it.
POOL = int(os.environ.get("RUACCENT_POOL", "2"))
THREADS = os.environ.get("RUACCENT_THREADS", "1")
PORT = int(os.environ.get("PORT", "8100"))

os.environ.setdefault("OMP_NUM_THREADS", THREADS)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("accent")


def prefetch() -> None:
    """Fetch the pinned models. Called at image build time, not at boot."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        local_dir=str(WORKDIR),
        allow_patterns=[
            "dictionary/**",
            "nn/nn_accent/**",
            "nn/nn_stress_usage_predictor/**",
            "nn/nn_yo_homograph_resolver/**",
            f"nn/nn_omograph/{MODEL_SIZE}/**",
        ],
    )
    # RUAccent's full mode imports Koziev's rule engine from inside its own
    # package directory, so that has to be fetched at the same pinned revision.
    snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        allow_patterns=["koziev/**"],
        local_dir=str(Path(ruaccent_package.__file__).resolve().parent),
    )


def build_pool() -> queue.Queue:
    pool: queue.Queue = queue.Queue()
    for i in range(POOL):
        acc = RUAccent()
        acc.load(
            omograph_model_size=MODEL_SIZE,
            use_dictionary=True,
            device="CPU",
            workdir=str(WORKDIR),
            tiny_mode=False,
        )
        pool.put(acc)
        log.info("accentuator %d/%d ready", i + 1, POOL)
    return pool


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool: queue.Queue

    def log_message(self, *_args):  # noqa: D401 - the default logger is too chatty
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/healthz", "/readyz"):
            self._send(200, {"status": "ok", "pool": POOL, "revision": REVISION})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/accent":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            texts = request.get("texts") or []
            skip = request.get("skip_regex") or r"[^\w\s]"
            if not isinstance(texts, list):
                raise ValueError("texts must be a list")
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return self._send(400, {"error": str(exc)})

        acc = self.pool.get()
        try:
            out = [acc.process_all(t, skip_regex=skip) if t.strip() else t for t in texts]
        except Exception as exc:  # noqa: BLE001
            log.exception("accentuation failed")
            return self._send(500, {"error": str(exc)})
        finally:
            self.pool.put(acc)
        self._send(200, {"texts": out})


def main() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        log.info("fetching RUAccent %s@%s", REPO, REVISION[:8])
        prefetch()
    else:
        log.info("using RUAccent %s@%s baked into the image", REPO, REVISION[:8])
    Handler.pool = build_pool()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    log.info("listening on 0.0.0.0:%d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
