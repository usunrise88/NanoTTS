#!/usr/bin/env python3
"""Produce warmed voices for a freshly exported bundle.

A bundle on its own has no voices, and a server with no voices cannot answer a
single request, so the installer runs this straight after the export.

Sources are whatever `get_state_for_audio_prompt` accepts: a local file, an
https:// URL, an `hf://repo/path[@rev]` path, an audio file of any format
upstream can read, or an upstream `.safetensors` state. Given none, this falls
back to the voice pocket-tts itself would pick for the checkpoint's language.

The state that comes back is in upstream's layout -- per-layer
`transformer.layers.N.self_attn/{offset,pad,cache}` -- so it goes through
pyrt's converter to reach the one this runtime loads.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parent))
from pyrt import XVibeRuntime  # noqa: E402


def flatten(state: dict, prefix: str = "") -> dict:
    """Nested module state -> the flat safetensors keys the converter reads.

    Upstream holds the cache as a tree of modules; on disk the same thing is
    `transformer.layers.0.self_attn/cache`, with a slash before the leaf.
    """
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            module, _, leaf = f"{prefix}{key}".rpartition(".")
            out[f"{module}/{leaf}"] = value.detach().cpu().numpy()
    return out


def slug(source: str) -> str:
    """A voice id from a source path: the file stem, minus the extension."""
    stem = Path(re.sub(r"[?#].*$", "", source.split("@")[0])).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()
    return stem or "voice"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--bundle", default="/out")
    ap.add_argument("--out", default="/voices")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("sources", nargs="*", help="audio or upstream .safetensors to warm")
    args = ap.parse_args()

    from pocket_tts.default_parameters import get_default_voice_for_language
    from pocket_tts.models.tts_model import TTSModel

    sources = list(args.sources)
    if not sources:
        # Predefined voices are states precomputed against the released weights,
        # so a custom checkpoint gets the audio file behind the fallback voice
        # instead -- which is what upstream does too.
        sources = [get_default_voice_for_language(
            args.language, config=args.config, checkpoint=None)]

    print(f"Loading {args.config or args.language} ...")
    model = TTSModel.load_model(config=args.config, language=args.language,
                                temp=0.5, eos_threshold=-4.0)
    model.eval()

    tokenizer = args.tokenizer or str(Path(args.bundle) / "tokenizer.model")
    rt = XVibeRuntime(args.bundle, tokenizer, precision="fp32")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = 0
    for source in sources:
        name = slug(source)
        try:
            state = model.get_state_for_audio_prompt(source, truncate=True)
        except Exception as exc:  # a missing voice must not fail the install
            print(f"  {name}: skipped ({exc})", file=sys.stderr)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            upstream = Path(tmp) / "upstream.safetensors"
            save_file(flatten(state), str(upstream))
            converted = rt.import_upstream_voice(upstream)

        dest = out / f"{name}.safetensors"
        rt.save_voice(converted, dest)
        # Written as root inside a container but read by the server as its own
        # user; a 0600 file here is indistinguishable from no voices at all.
        dest.chmod(0o644)
        frames = int(converted["step"][0])
        print(f"  {name}.safetensors  {frames} frames "
              f"({frames / rt.frame_rate:.1f} s of reference)")
        made += 1

    if not made:
        print("No voices could be produced; upload one through POST /v1/voices.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
