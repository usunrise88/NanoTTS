#!/usr/bin/env python3
"""Load and quality harness for the NanoTTS server.

Measures what actually matters in production: TTFB and RTF percentiles *under
concurrency*, not a median of five runs on an idle box. Optionally transcribes
every generated utterance through an OpenAI-compatible ASR endpoint and reports
WER/CER, which is the gate any optimisation has to pass before it lands.

  python bench.py --url http://localhost:8099 --voice male_deep --concurrency 4
  python bench.py --url ... --asr-url http://host:8080/v1 --asr-key KEY --asr-model NAME
"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import wave

ACUTE = "́"


def normalise(text: str) -> list[str]:
    """Strip stress marks, case and punctuation so WER compares words, not typography."""
    t = unicodedata.normalize("NFD", text.lower())
    t = t.replace(ACUTE, "").replace("+", "")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return t.split()


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def post_json(url: str, payload: dict, key: str | None, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.perf_counter()
    first = None
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # The first read asks for a single byte on purpose. read(8192) blocks
        # until the buffer fills, and the first audio chunk is smaller than
        # that, so a large first read measures the *second* chunk's arrival and
        # inflates TTFB by a whole chunk.
        head = r.read(1)
        if head:
            first = time.perf_counter() - t0
            chunks.append(head)
            while True:
                block = r.read(65536)
                if not block:
                    break
                chunks.append(block)
    return b"".join(chunks), first or 0.0, time.perf_counter() - t0


def wav_seconds(data: bytes) -> float:
    """Length in seconds, trusting the byte count over the header.

    A streamed WAV cannot know its length when the header goes out, so the size
    fields carry 0xFFFFFFFF and `wave` happily reports a few million frames.
    """
    if len(data) < 44:
        return 0.0
    try:
        with wave.open(io.BytesIO(data)) as w:
            frames, rate, width, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
            by_bytes = (len(data) - 44) / max(1, width * ch)
            if frames <= 0 or frames > by_bytes * 1.01:
                frames = by_bytes
            return frames / rate
    except Exception:
        return max(0.0, (len(data) - 44) / 2 / 24000)


def transcribe(asr_url: str, key: str, model: str, wav: bytes, timeout: float) -> str:
    boundary = "----nanottsbench"
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n")
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    )
    head = "".join(parts).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        asr_url.rstrip("/") + "/audio/transcriptions",
        data=head + wav + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("text", "")


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8099")
    ap.add_argument("--voice", default="male_deep")
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(__file__), "texts_ru.txt"))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asr-url", default=None)
    ap.add_argument("--asr-key", default=os.environ.get("ASR_KEY"))
    ap.add_argument("--asr-model", default="gigaam-v3-rnnt-punct-ru@2025-12-16")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    with open(args.texts, encoding="utf-8") as f:
        texts = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    jobs = [(i, texts[i % len(texts)]) for i in range(len(texts) * args.repeat)]
    print(f"{len(jobs)} utterances, concurrency {args.concurrency}, voice {args.voice}")

    endpoint = args.url.rstrip("/") + "/v1/audio/speech"

    def run(job):
        idx, text = job
        payload = {"input": text, "voice": args.voice, "response_format": "wav",
                   "nanotts": {"seed": args.seed + idx if args.seed else 0}}
        try:
            data, ttfb, total = post_json(endpoint, payload, args.api_key, args.timeout)
            return {"idx": idx, "text": text, "wav": data, "ttfb": ttfb, "total": total,
                    "seconds": wav_seconds(data)}
        except (urllib.error.URLError, TimeoutError) as e:
            return {"idx": idx, "text": text, "error": str(e)}

    t_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, jobs))
    wall = time.perf_counter() - t_start

    ok = [r for r in results if "wav" in r]
    failed = [r for r in results if "wav" not in r]
    ttfbs = [r["ttfb"] * 1000 for r in ok]
    rtfs = [r["total"] / r["seconds"] for r in ok if r["seconds"] > 0]
    audio = sum(r["seconds"] for r in ok)

    print(f"\nok {len(ok)}/{len(results)}   failed {len(failed)}")
    print(f"wall {wall:.1f}s   audio {audio:.1f}s   throughput {audio / wall:.2f}x realtime")
    print(f"TTFB ms   p50 {pct(ttfbs, .5):.0f}   p95 {pct(ttfbs, .95):.0f}   p99 {pct(ttfbs, .99):.0f}"
          f"   max {max(ttfbs, default=0):.0f}")
    print(f"RTF       p50 {pct(rtfs, .5):.3f}   p95 {pct(rtfs, .95):.3f}   "
          f"mean {statistics.mean(rtfs) if rtfs else 0:.3f}")
    for r in failed[:5]:
        print("  failed:", r.get("error"))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for r in ok:
            with open(os.path.join(args.save_dir, f"{r['idx']:04d}.wav"), "wb") as f:
                f.write(r["wav"])

    if not args.asr_url:
        return 0

    print("\ntranscribing ...")
    def score(r):
        try:
            hyp = transcribe(args.asr_url, args.asr_key, args.asr_model, r["wav"], args.timeout)
        except Exception as e:
            return {"error": str(e)}
        ref_w, hyp_w = normalise(r["text"]), normalise(hyp)
        ref_c, hyp_c = list(" ".join(ref_w)), list(" ".join(hyp_w))
        return {"ref": r["text"], "hyp": hyp,
                "werr": edit_distance(ref_w, hyp_w), "wn": len(ref_w),
                "cerr": edit_distance(ref_c, hyp_c), "cn": len(ref_c)}

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        scored = list(pool.map(score, ok))

    good = [s for s in scored if "wn" in s]
    if not good:
        print("no transcriptions returned:", scored[:2])
        return 1
    wer = sum(s["werr"] for s in good) / max(1, sum(s["wn"] for s in good))
    cer = sum(s["cerr"] for s in good) / max(1, sum(s["cn"] for s in good))
    perfect = sum(1 for s in good if s["werr"] == 0)
    print(f"WER {wer * 100:.2f}%   CER {cer * 100:.2f}%   clean {perfect}/{len(good)}")
    for s in sorted(good, key=lambda s: -s["werr"])[:5]:
        if s["werr"]:
            print(f"  -{s['werr']:2d}  ref: {s['ref']}\n       hyp: {s['hyp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
