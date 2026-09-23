"""End-to-end: precomputed NanoTTS voice -> ONNX pipeline -> WAV."""
import time, numpy as np
from pyrt import XVibeRuntime, write_wav

rt = XVibeRuntime("/out", "/assets/weights/tokenizer.model", threads=4)
voice = rt.import_upstream_voice("/assets/voices/voice_3_male_deep.safetensors")
print("voice step (prefix frames):", voice["step"].tolist())

text = "Прив+ет! Это пров+ерка р+усской р+ечи на серверном процессоре."
t0 = time.time()
audio = rt.generate(text, voice, seed=42)
dt = time.time() - t0
dur = len(audio) / rt.sample_rate
print(f"samples={len(audio)}  duration={dur:.2f}s  wall={dt:.2f}s  RTF={dt/max(dur,1e-9):.3f}"
      f"  ({dur/max(dt,1e-9):.2f}x realtime)")
print(f"peak={np.abs(audio).max():.3f}  rms={np.sqrt((audio**2).mean()):.4f}  finite={np.isfinite(audio).all()}")
write_wav("/out/e2e_male_deep.wav", audio, rt.sample_rate)
print("wrote /out/e2e_male_deep.wav")
