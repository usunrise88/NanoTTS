"""Accent the benchmark set with RUAccent, pinned exactly as GenVoice pins it.

Exists to answer one question with a measurement rather than an opinion: how
much of the quality gap comes from the server not placing stress at all.
"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import ruaccent as pkg
from ruaccent import RUAccent

REPO = "ruaccent/accentuator"
REV = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855"
SIZE = "turbo3.1"

work = Path("/tmp/ruaccent")
work.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=REPO, revision=REV, local_dir=str(work), allow_patterns=[
    "dictionary/**", "nn/nn_accent/**", "nn/nn_stress_usage_predictor/**",
    "nn/nn_yo_homograph_resolver/**", f"nn/nn_omograph/{SIZE}/**"])
snapshot_download(repo_id=REPO, revision=REV, allow_patterns=["koziev/**"],
                  local_dir=str(Path(pkg.__file__).resolve().parent))

acc = RUAccent()
acc.load(omograph_model_size=SIZE, use_dictionary=True, device="CPU",
         workdir=str(work), tiny_mode=False)

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
out = []
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        out.append(line); continue
    out.append(acc.process_all(line, skip_regex=r"[^\w\s]"))
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"  расставлено ударений в {len([l for l in out if l and not l.startswith('#')])} строках")
print("  пример:", out[1][:110])
