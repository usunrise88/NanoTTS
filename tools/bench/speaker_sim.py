#!/usr/bin/env python3
"""Speaker similarity in the model's own conditioning space.

mimi_encoder already ends in speaker_proj, so its output is exactly the space
the FlowLM is conditioned on. Mean-pooling it and comparing by cosine asks the
only question that matters here: does the synthesised audio land where the
reference does, as far as this model is concerned.
"""
import sys, wave, math, array
import numpy as np
import onnxruntime as ort

SR, FRAME = 24000, 1920

def read_wav(path):
    with open(path, 'rb') as f:
        d = f.read()
    import struct
    off, fmt, data = 12, None, None
    while off + 8 <= len(d):
        cid, sz = d[off:off+4], struct.unpack('<I', d[off+4:off+8])[0]
        if cid == b'fmt ': fmt = struct.unpack('<HHIIHH', d[off+8:off+8+16])
        if cid == b'data': data = d[off+8:off+8+sz]
        off += 8 + sz + (sz & 1)
    tag, ch, rate, _, _, bits = fmt
    if bits == 16:  a = np.frombuffer(data, '<i2').astype(np.float32) / 32768
    elif bits == 32 and tag == 3: a = np.frombuffer(data, '<f4').astype(np.float32)
    elif bits == 32: a = np.frombuffer(data, '<i4').astype(np.float32) / 2147483648
    elif bits == 24:
        raw = np.frombuffer(data[: len(data) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        a = np.where(v & 0x800000, v - 0x1000000, v).astype(np.float32) / 8388608
    else: raise SystemExit(f'unsupported: {bits} bit tag {tag}')
    if ch > 1: a = a.reshape(-1, ch).mean(1)
    if rate != SR:  # linear is plenty for a similarity probe
        n = int(len(a) * SR / rate)
        a = np.interp(np.arange(n) * rate / SR, np.arange(len(a)), a).astype(np.float32)
    return a

def embed(sess, audio):
    pad = (-len(audio)) % FRAME
    if pad: audio = np.concatenate([audio, np.zeros(pad, np.float32)])
    cond = sess.run(None, {'audio': audio.reshape(1, 1, -1)})[0][0]   # [frames, 1024]
    v = cond.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)

sess = ort.InferenceSession('/bundle/mimi_encoder.onnx', providers=['CPUExecutionProvider'])
ref = embed(sess, read_wav(sys.argv[1]))
print(f"{'файл':34} {'cos к референсу':>16}")
for p in sys.argv[2:]:
    v = embed(sess, read_wav(p))
    print(f"{p.split('/')[-1]:34} {float(ref @ v):>16.4f}")
