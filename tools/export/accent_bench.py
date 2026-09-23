"""How much latency accentuation would add if it sat in front of the synth."""
import time, statistics
from pathlib import Path
from huggingface_hub import snapshot_download
import ruaccent as pkg
from ruaccent import RUAccent

REPO, REV, SIZE = "ruaccent/accentuator", "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855", "turbo3.1"
work = Path("/tmp/ruaccent"); work.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=REPO, revision=REV, local_dir=str(work), allow_patterns=[
    "dictionary/**", "nn/nn_accent/**", "nn/nn_stress_usage_predictor/**",
    "nn/nn_yo_homograph_resolver/**", f"nn/nn_omograph/{SIZE}/**"])
snapshot_download(repo_id=REPO, revision=REV, allow_patterns=["koziev/**"],
                  local_dir=str(Path(pkg.__file__).resolve().parent))

t0 = time.perf_counter()
acc = RUAccent()
acc.load(omograph_model_size=SIZE, use_dictionary=True, device="CPU",
         workdir=str(work), tiny_mode=False)
print(f"  загрузка модели: {(time.perf_counter()-t0):.1f} с (разово, при старте)")

lines = [l for l in Path("/bench/texts_long_ru.txt").read_text(encoding="utf-8").splitlines()
         if l.strip() and not l.startswith("#")]
short = [l for l in Path("/bench/texts_ru_plain.txt").read_text(encoding="utf-8").splitlines()
         if l.strip() and not l.startswith("#")]

for name, data in (("длинные", lines), ("короткие", short)):
    acc.process_all(data[0], skip_regex=r"[^\w\s]")          # прогрев
    times = []
    for l in data:
        t = time.perf_counter()
        acc.process_all(l, skip_regex=r"[^\w\s]")
        times.append((time.perf_counter()-t)*1000)
    times.sort()
    print(f"  {name:9} n={len(times):2}  медиана {statistics.median(times):6.1f} мс  "
          f"p95 {times[int(len(times)*0.95)-1]:6.1f} мс  макс {max(times):6.1f} мс")
