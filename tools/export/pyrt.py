#!/usr/bin/env python3
"""Reference ONNX runtime for the NanoTTS bundle.

This exists to pin down behaviour before any of it is written in C++: the AR
loop, the LSD solver, voice warm-up, the text pipeline and the state layout are
all here in a form that can be diffed against the torch pipeline. The C++
runtime is expected to reproduce this file's output sample for sample (within
float tolerance), so treat it as the spec rather than as a utility.
"""

import json
import math
import re
import unicodedata
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file, save_file

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
ACUTE = "́"
_PLUS_VOWEL = re.compile(rf"\+([{VOWELS}])")
_PLUS_YO = re.compile(r"\+([ёЁ])")
_YO_ACUTE = re.compile(rf"([ёЁ]){ACUTE}")


# ---------------------------------------------------------------- text


def to_model_stress(text: str) -> str:
    """`+о` -> `о` + U+0301. Marks on ё are dropped: ё is inherently stressed and
    an extra mark pushes the tokenizer out of distribution."""
    t = unicodedata.normalize("NFC", text)
    t = _PLUS_YO.sub(r"\1", t)
    t = _YO_ACUTE.sub(r"\1", t)
    t = _PLUS_VOWEL.sub(rf"\1{ACUTE}", t)
    return unicodedata.normalize("NFC", t)


def prepare_text_prompt(text: str, pad_with_spaces: bool, remove_semicolons: bool):
    """Byte-for-byte port of pocket_tts.models.tts_model.prepare_text_prompt.

    The single-pass `replace("  ", " ")` is not idempotent; that is upstream
    behaviour and changing it would change the chunking.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty text")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    frames_after_eos = 3 if len(text.split()) <= 4 else 1
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text = text + "."
    if pad_with_spaces and len(text.split()) < 5:
        text = " " * 8 + text
    return text, frames_after_eos


def _boundaries(tokens, boundary_set):
    idx, prev = [0], False
    for i, t in enumerate(tokens):
        if t in boundary_set:
            prev = True
        else:
            if prev:
                idx.append(i)
            prev = False
    idx.append(len(tokens))
    return idx


def _segments(tokens, bounds, sp):
    return [
        (bounds[i + 1] - bounds[i], sp.decode(tokens[bounds[i] : bounds[i + 1]]))
        for i in range(len(bounds) - 1)
    ]


def split_chunks(sp, text, max_tokens, pad_with_spaces, remove_semicolons):
    """Port of split_into_best_sentences: sentence split, comma fallback for
    oversized sentences, then greedy merge up to `max_tokens`."""
    text, _ = prepare_text_prompt(text, pad_with_spaces, remove_semicolons)
    tokens = sp.encode(text.strip())
    eos_set = set(sp.encode(".!...?")[1:])
    segs = _segments(tokens, _boundaries(tokens, eos_set), sp)

    fallback = set(sp.encode(",;:")[1:])
    refined = []
    for n, seg in segs:
        if n <= max_tokens:
            refined.append((n, seg))
            continue
        sub = sp.encode(seg.strip())
        parts = _segments(sub, _boundaries(sub, fallback), sp)
        refined.extend(parts if len(parts) > 1 else [(n, seg)])

    chunks, cur, cur_n = [], "", 0
    for n, seg in refined:
        if not cur:
            cur, cur_n = seg, n
        elif cur_n + n > max_tokens:
            chunks.append(cur.strip())
            cur, cur_n = seg, n
        else:
            cur += " " + seg
            cur_n += n
    if cur:
        chunks.append(cur.strip())
    return chunks


# ---------------------------------------------------------------- runtime


class XVibeRuntime:
    def __init__(self, bundle_dir, tokenizer_path, precision="fp32", threads=4):
        self.dir = Path(bundle_dir)
        self.meta = json.loads((self.dir / "bundle.json").read_text())
        self.sample_rate = self.meta["sample_rate"]
        self.latent_dim = self.meta["latent_dim"]
        self.cond_dim = self.meta["conditioning_dim"]
        self.nl = self.meta["num_layers"]
        self.H = self.meta["num_heads"]
        self.Dh = self.meta["dim_per_head"]
        self.max_seq = self.meta["max_seq"]
        self.samples_per_frame = self.meta["samples_per_frame"]
        self.frame_rate = self.meta["frame_rate"]

        import sentencepiece

        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def sess(name, quantizable=True):
            sfx = "_int8" if (precision == "int8" and quantizable) else ""
            path = self.dir / f"{name}{sfx}.onnx"
            if not path.exists():
                path = self.dir / f"{name}.onnx"
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        self.text_cond = sess("text_conditioner", False)
        self.flow_main = sess("flow_lm_main")
        self.flow_step = sess("flow_lm_flow")
        self.mimi_enc = sess("mimi_encoder", False)
        self.mimi_dec = sess("mimi_decoder", False)
        self.mimi_state_meta = self.meta["mimi_state"]

    # -- state helpers ------------------------------------------------

    def empty_flow_state(self, batch=1):
        state = {"step": np.zeros(batch, dtype=np.int64)}
        for i in range(self.nl):
            for kv in ("k", "v"):
                state[f"cache_{kv}_{i}"] = np.zeros(
                    (batch, self.max_seq, self.H, self.Dh), dtype=np.float16
                )
        return state

    def empty_mimi_state(self, batch=1):
        dt = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "bool": np.bool_}
        out = {}
        for e in self.mimi_state_meta:
            shape = list(e["shape"])
            shape[0] = batch
            out[e["name"]] = np.zeros(shape, dtype=dt[e["dtype"]])
        return out

    def _run_main(self, sequence, text_emb, state):
        feed = {"sequence": sequence, "text_embeddings": text_emb, **state}
        out = self.flow_main.run(None, feed)
        names = [o.name for o in self.flow_main.get_outputs()]
        new_state = {n[4:]: v for n, v in zip(names, out) if n.startswith("out_")}
        return out[0], out[1], new_state

    # -- voices -------------------------------------------------------

    def warm_voice(self, audio: np.ndarray):
        """Reference audio -> flow state. Audio must be mono float32 at
        `sample_rate`; it is padded up to a whole frame because the encoder
        graph was traced frame-aligned."""
        n = len(audio)
        pad = (-n) % self.samples_per_frame
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
        cond = self.mimi_enc.run(None, {"audio": audio.reshape(1, 1, -1)})[0]
        if self.meta["insert_bos_before_voice"]:
            bos = np.load(self.dir / "bos_before_voice.npy") if (
                self.dir / "bos_before_voice.npy"
            ).exists() else None
            if bos is not None:
                cond = np.concatenate([bos.astype(np.float32), cond], axis=1)
        state = self.empty_flow_state(1)
        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        _, _, state = self._run_main(empty_seq, cond.astype(np.float32), state)
        return state

    def save_voice(self, state, path):
        save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(path))

    def load_voice(self, path):
        return {k: v for k, v in load_file(str(path)).items()}

    def import_upstream_voice(self, path):
        """Convert a genvoice/xVibePocketTTS voice into this bundle's layout.

        Upstream stores `transformer.layers.N.self_attn/{offset,pad,cache}` with
        cache [2,1,T,16,64] fp32; here K and V live in separate fp16 buffers of
        capacity max_seq and the position is one shared vector.
        """
        raw = load_file(str(path))
        state = self.empty_flow_state(1)
        step = None
        for i in range(self.nl):
            base = f"transformer.layers.{i}.self_attn"
            cache = raw[f"{base}/cache"]  # [2, 1, T, H, Dh]
            T = cache.shape[2]
            assert T <= self.max_seq, f"voice prefix {T} exceeds max_seq {self.max_seq}"
            state[f"cache_k_{i}"][:, :T] = cache[0].astype(np.float16)
            state[f"cache_v_{i}"][:, :T] = cache[1].astype(np.float16)
            off = raw[f"{base}/offset"].reshape(-1)[0]
            step = int(off) if step is None else step
            assert int(off) == step, "layers disagree on offset"
        state["step"] = np.array([step], dtype=np.int64)
        return state

    # -- generation ---------------------------------------------------

    def generate(self, text, voice_state, temperature=None, eos_threshold=None,
                 lsd_steps=None, seed=None, chunk_frames=15, first_chunk_frames=2,
                 max_frames=500):
        d = self.meta["defaults"]
        temperature = d["temperature"] if temperature is None else temperature
        eos_threshold = d["eos_threshold"] if eos_threshold is None else eos_threshold
        lsd_steps = d["lsd_steps"] if lsd_steps is None else lsd_steps
        rng = np.random.default_rng(seed)

        chunks = split_chunks(
            self.sp,
            to_model_stress(text),
            self.meta["max_token_per_chunk"],
            self.meta["pad_with_spaces_for_short_inputs"],
            self.meta["remove_semicolons"],
        )

        audio_out = []
        for chunk in chunks:
            prepared, guess = prepare_text_prompt(
                chunk,
                self.meta["pad_with_spaces_for_short_inputs"],
                self.meta["remove_semicolons"],
            )
            frames_after_eos = guess + 2
            ids = np.array([self.sp.encode(prepared)], dtype=np.int64)
            text_emb = self.text_cond.run(None, {"token_ids": ids})[0].astype(np.float32)
            if text_emb.ndim == 2:
                text_emb = text_emb[None]

            state = {k: v.copy() for k, v in voice_state.items()}
            empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
            empty_text = np.zeros((1, 0, self.cond_dim), dtype=np.float32)
            _, _, state = self._run_main(empty_seq, text_emb, state)

            latents = self._ar_loop(
                state, empty_text, temperature, eos_threshold, lsd_steps,
                rng, frames_after_eos, max_frames,
            )
            if latents:
                audio_out.append(self._decode(latents, chunk_frames, first_chunk_frames))

        return np.concatenate(audio_out) if audio_out else np.zeros(0, dtype=np.float32)

    def _ar_loop(self, state, empty_text, temperature, eos_threshold, lsd_steps,
                 rng, frames_after_eos, max_frames):
        cur = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        dt = 1.0 / lsd_steps
        st = [
            (np.array([[j / lsd_steps]], np.float32), np.array([[(j + 1) / lsd_steps]], np.float32))
            for j in range(lsd_steps)
        ]
        std = math.sqrt(temperature) if temperature > 0 else 0.0
        latents, eos_at = [], None

        for step in range(max_frames):
            cond, eos_logit, state = self._run_main(cur, empty_text, state)
            if eos_at is None and eos_logit[0][0] > eos_threshold:
                eos_at = step
            if eos_at is not None and step >= eos_at + frames_after_eos:
                break

            x = (rng.normal(0.0, std, (1, self.latent_dim)).astype(np.float32)
                 if std > 0 else np.zeros((1, self.latent_dim), np.float32))
            for s, t in st:
                x = x + self.flow_step.run(None, {"c": cond, "s": s, "t": t, "x": x})[0] * dt

            cur = x.reshape(1, 1, self.latent_dim)
            latents.append(cur)
        return latents

    def _decode(self, latents, chunk_frames, first_chunk_frames):
        state = self.empty_mimi_state(1)
        names = [o.name for o in self.mimi_dec.get_outputs()]
        out, i, want = [], 0, first_chunk_frames
        while i < len(latents):
            take = min(want, len(latents) - i)
            block = np.concatenate(latents[i : i + take], axis=1)
            res = self.mimi_dec.run(None, {"latent": block, **state})
            out.append(res[0].reshape(-1))
            state = {n[4:]: v for n, v in zip(names, res) if n.startswith("out_")}
            i += take
            want = chunk_frames
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def write_wav(path, audio, sample_rate):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
