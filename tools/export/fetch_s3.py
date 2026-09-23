#!/usr/bin/env python3
"""Builds a bundle from a checkpoint that already ships ONNX.

Supertonic 3 and TeraTTS v2 are released as graphs rather than weights, so there
is nothing to trace: the work is downloading them, converting the voices into
this runtime's voice files, and writing the manifest that tells the server which
backend to load and how this particular checkpoint wants its text.

The two share an architecture -- the graph interfaces are identical down to the
tensor shapes, only an output name differs -- but not their text conventions:
TeraTTS was trained with literal <ru>...</ru> spans and a 134-character
vocabulary with no digits, Supertonic with 8,321 characters including digits and
no language tags. Those differences live in the manifest, not in code.

  python fetch_s3.py --model TeraSpace/TeraTTSv2 --out /out --voices /voices
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.numpy import save_file

# What each known release calls its files, and what its text layer needs. A repo
# that is not listed still works if it follows one of these layouts; the probe
# below picks whichever one its file list matches.
LAYOUTS = {
    "teratts": {
        "graph_dir": "models",
        "indexer": "unicode_indexer.json",
        "styles_dir": "styles",
        "style_format": "npy",
        "sampler": ["sampler_distilled_cfg3_8step.onnx", "sampler_teacher_8step.onnx"],
        "text": {"language_tags": True, "default_language": "ru"},
        "sample_rate": 44100,
        "speed": 1.05,
    },
    "supertonic": {
        "graph_dir": "onnx",
        "indexer": "onnx/unicode_indexer.json",
        "styles_dir": "voice_styles",
        "style_format": "json",
        "sampler": ["vector_estimator.onnx"],
        "text": {"language_tags": False, "default_language": "en"},
        "sample_rate": 44100,
        "speed": 1.0,
    },
}

SAMPLES_PER_FRAME = 3072
VOCODER_CONTEXT_FRAMES = 20
STREAM_CHUNK_FRAMES = 16


def detect_layout(files: list[str]) -> str:
    if any(f.startswith("styles/") for f in files):
        return "teratts"
    if any(f.startswith("voice_styles/") for f in files):
        return "supertonic"
    raise SystemExit("cannot tell which layout this repo uses: no styles/ or voice_styles/")


def fetch(repo: str, name: str, out: Path, revision: str | None) -> Path:
    src = hf_hub_download(repo_id=repo, filename=name, revision=revision)
    dst = out / Path(name).name
    shutil.copyfile(src, dst)
    return dst


def load_style(repo: str, files: list[str], voice: str, layout: dict, revision: str | None):
    """Returns (style_ttl, style_dp) as float32 arrays, whichever way they ship."""
    base = f"{layout['styles_dir']}/{voice}"
    if layout["style_format"] == "npy":
        ttl = np.load(hf_hub_download(repo_id=repo, filename=f"{base}/style_ttl.npy", revision=revision))
        dp = np.load(hf_hub_download(repo_id=repo, filename=f"{base}/style_dp.npy", revision=revision))
    else:
        raw = json.loads(Path(hf_hub_download(repo_id=repo, filename=f"{base}.json", revision=revision)).read_text())
        # The JSON form stores flat lists plus their shapes.
        def pick(*names):
            for n in names:
                if n in raw:
                    return raw[n]
            raise SystemExit(f"{base}.json has none of {names}; keys: {list(raw)[:8]}")
        ttl = np.asarray(pick("style_ttl", "ttl", "style"), dtype=np.float32)
        dp = np.asarray(pick("style_dp", "dp"), dtype=np.float32)
        if ttl.ndim == 1:
            ttl = ttl.reshape(1, -1, 256)
        if dp.ndim == 1:
            dp = dp.reshape(1, 8, 16)
    return np.ascontiguousarray(ttl, dtype=np.float32), np.ascontiguousarray(dp, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HuggingFace repo, e.g. TeraSpace/TeraTTSv2")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default="/out", help="bundle directory")
    ap.add_argument("--voices", default=None, help="where to write the voice files")
    ap.add_argument("--sampler", default=None, help="which sampler graph to take")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(args.model, revision=args.revision)
    kind = detect_layout(files)
    layout = LAYOUTS[kind]
    print(f"{args.model}: {kind} layout")

    gd = layout["graph_dir"]
    sampler_name = args.sampler
    if sampler_name is None:
        for candidate in layout["sampler"]:
            if f"{gd}/{candidate}" in files:
                sampler_name = candidate
                break
    if sampler_name is None:
        raise SystemExit(f"no sampler graph found; looked for {layout['sampler']}")

    # The runtime asks the bundle for graphs by role, so they are stored under
    # role names and the release's own filenames stop mattering here.
    wanted = {
        "text_encoder.onnx": f"{gd}/text_encoder.onnx",
        "duration_predictor.onnx": f"{gd}/duration_predictor.onnx",
        "sampler.onnx": f"{gd}/{sampler_name}",
        "vocoder.onnx": f"{gd}/vocoder.onnx",
    }
    for local, remote in wanted.items():
        if remote not in files:
            raise SystemExit(f"{args.model} has no {remote}")
        src = hf_hub_download(repo_id=args.model, filename=remote, revision=args.revision)
        shutil.copyfile(src, out / local)
        print(f"  {local:26s} {(out / local).stat().st_size / 1e6:7.1f} MB")

    indexer_src = hf_hub_download(repo_id=args.model, filename=layout["indexer"], revision=args.revision)
    shutil.copyfile(indexer_src, out / "unicode_indexer.json")
    table = json.loads((out / "unicode_indexer.json").read_text())
    supported = sum(1 for x in table if x >= 0)
    has_digits = all(table[ord(d)] >= 0 for d in "0123456789")
    print(f"  unicode_indexer.json       {supported} characters, digits: {'yes' if has_digits else 'no'}")

    # Voices come out as this runtime's own files so the server loads them the
    # same way it loads a warmed Pocket voice.
    voice_dir = Path(args.voices) if args.voices else out / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    if layout["style_format"] == "npy":
        names = sorted({f.split("/")[1] for f in files if f.startswith(layout["styles_dir"] + "/")})
    else:
        names = sorted(
            Path(f).stem for f in files if f.startswith(layout["styles_dir"] + "/") and f.endswith(".json")
        )
    ttl_shape = dp_shape = None
    for voice in names:
        try:
            ttl, dp = load_style(args.model, files, voice, layout, args.revision)
        except Exception as exc:  # one odd voice must not sink the whole fetch
            print(f"  {voice}: skipped ({exc})", file=sys.stderr)
            continue
        dest = voice_dir / f"{voice}.safetensors"
        save_file({"style_ttl": ttl, "style_dp": dp}, str(dest))
        dest.chmod(0o644)
        ttl_shape, dp_shape = list(ttl.shape), list(dp.shape)
        print(f"  voice {voice:24s} ttl {tuple(ttl.shape)}  dp {tuple(dp.shape)}")
    if ttl_shape is None:
        raise SystemExit("no voices could be read; the server would have nothing to speak with")

    manifest = {
        "schema_version": 1,
        "architecture": "s3",
        "bundle_name": args.model.split("/")[-1],
        "sample_rate": layout["sample_rate"],
        "samples_per_frame": SAMPLES_PER_FRAME,
        "frame_rate": layout["sample_rate"] / SAMPLES_PER_FRAME,
        "s3": {
            "latent_dim": 144,
            "vocoder_context_frames": VOCODER_CONTEXT_FRAMES,
            "stream_chunk_frames": STREAM_CHUNK_FRAMES,
            "speed": layout["speed"],
            "guidance": 3.0,
            "style_ttl_shape": ttl_shape,
            "style_dp_shape": dp_shape,
            "language_tags": layout["text"]["language_tags"],
            "default_language": layout["text"]["default_language"],
        },
        "defaults": {"temperature": 0.5, "eos_threshold": -4.0, "lsd_steps": 1},
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'bundle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
