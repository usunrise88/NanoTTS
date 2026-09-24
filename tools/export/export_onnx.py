#!/usr/bin/env python3
"""Export genvoice/xVibePocketTTS to ONNX for the NanoTTS C++ runtime.

Five stateless graphs — all recurrent state travels through inputs/outputs:

    flow_lm_main      AR backbone, split K/V cache (fp16), EOS head
    flow_lm_flow      stateless LSD step: (c, s, t, x) -> flow_dir
    text_conditioner  token ids -> embeddings
    mimi_encoder      audio -> 1024-d conditioning (speaker_proj baked in)
    mimi_decoder      latent + state -> audio

Differences from VolgaGerm/PocketTTS.cpp's export_onnx.py, which this is derived from:

  * `step` is per-row (int64 [B]) instead of a shared scalar, and RoPE takes a
    per-row offset. Rows are left-aligned and independent, so slots may sit at
    different positions in the same batch. That is what makes real continuous
    batching possible rather than lockstep cohorts, and it removes the need for
    upstream's `pad` right-alignment trick entirely.
  * One shared `step` state instead of one per layer: 13 state tensors rather
    than 18, so the runtime binds fewer buffers per AR step.
  * Batch axis is dynamic on every graph.
  * Layer geometry is read from the checkpoint instead of being hardcoded.
"""

import argparse
import json
import math
import os
from pathlib import Path

# The runtime type-checker rejects the tensors the tracer substitutes for
# plain ints, so it has to go before pocket_tts is imported.
os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")

import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.models.tts_model import TTSModel

import mimi_export

DEFAULT_MAX_SEQ = 512  # FlowLM KV capacity, in positions (voice + text + frames)
MIMI_CACHE_LEN = 250  # matches the Mimi transformer `context`


# --------------------------------------------------------------------------
# RoPE with a per-row offset
# --------------------------------------------------------------------------


def apply_rope_batched(
    q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor, max_period: float
):
    """Same rotation as pocket_tts.modules.rope.apply_rope, but `offset` is [B].

    Reduces to the upstream result when every row carries the same offset.
    """
    B, T, H, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))  # [D/2]

    ts = torch.arange(T, device=q.device, dtype=torch.float32).view(1, T)
    ts = ts + offset.to(torch.float32).view(B, 1)  # [B, T]
    angles = ts.view(B, T, 1, 1) * freqs.view(1, 1, 1, -1)  # [B, T, 1, D/2]
    rotr, roti = torch.cos(angles), torch.sin(angles)

    qv = q.view(B, T, H, D // 2, 2)
    kv = k.view(B, T, H, D // 2, 2)
    qr, qi = qv[..., 0].float(), qv[..., 1].float()
    kr, ki = kv[..., 0].float(), kv[..., 1].float()

    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1)
    return qo.view(B, T, H, D).to(q.dtype), ko.view(B, T, H, D).to(k.dtype)


# --------------------------------------------------------------------------
# flow_lm_main
# --------------------------------------------------------------------------


class FlowLMMainWrapper(nn.Module):
    """FlowLM backbone with an explicit split K/V cache.

    Inputs
        sequence        [B, S, ldim] fp32 — NaN marks BOS (the upstream protocol)
        text_embeddings [B, T, dim]  fp32 — voice or text prefix; empty on AR steps
        step            [B]          int64 — per-row write position
        cache_k_{i}     [B, MAX, H, Dh] fp16
        cache_v_{i}     [B, MAX, H, Dh] fp16

    Outputs
        conditioning [B, dim], eos_logit [B, 1], out_step [B],
        out_cache_k_{i}, out_cache_v_{i}
    """

    def __init__(self, flow_lm, max_seq: int):
        super().__init__()
        self.max_seq = max_seq
        self.bos_emb = flow_lm.bos_emb
        self.input_linear = flow_lm.input_linear
        self.out_norm = flow_lm.out_norm
        self.out_eos = flow_lm.out_eos
        self.layers = flow_lm.transformer.layers
        self.rope_max_period = float(flow_lm.transformer.rope.max_period)

        attn0 = self.layers[0].self_attn
        self.num_layers = len(self.layers)
        self.num_heads = attn0.num_heads
        self.dim_per_head = attn0.dim_per_head
        self.d_model = attn0.embed_dim
        assert attn0.context is None, "sliding-window context is not supported by this export"

    def _attention(self, attn, x, cache_k, cache_v, step):
        B, L, _ = x.shape
        H, Dh = self.num_heads, self.dim_per_head

        projected = attn.in_proj(x)
        q, k, v = torch.unbind(projected.view(B, L, 3, H, Dh), dim=2)
        q, k = apply_rope_batched(q, k, step, self.rope_max_period)

        # Scatter the new K/V at each row's own position.
        positions = step.view(B, 1) + torch.arange(L, device=x.device, dtype=torch.long).view(1, L)
        idx = positions.view(B, L, 1, 1).expand(B, L, H, Dh)
        updated_k = cache_k.scatter(1, idx, k.half())
        updated_v = cache_v.scatter(1, idx, v.half())

        # Attend only over slots any row has actually written. Rows that lag
        # behind are cut off by the causal mask below, and unwritten slots hold
        # zeros (never NaN), so they cannot poison the softmax.
        valid_len = step.max() + L
        valid_k = updated_k[:, :valid_len].float()
        valid_v = updated_v[:, :valid_len].float()

        pos_k = torch.arange(valid_len, device=x.device, dtype=torch.long).view(1, 1, -1)
        pos_q = step.view(B, 1) + torch.arange(L, device=x.device, dtype=torch.long).view(1, L)
        allowed = (pos_q.unsqueeze(-1) - pos_k) >= 0  # [B, L, valid]
        attn_mask = torch.log(allowed.float()).unsqueeze(1)  # [B, 1, L, valid]

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), valid_k.transpose(1, 2), valid_v.transpose(1, 2), attn_mask
        )
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return attn.out_proj(out), updated_k, updated_v

    def forward(self, sequence, text_embeddings, step, *caches):
        k_caches = [caches[2 * i] for i in range(self.num_layers)]
        v_caches = [caches[2 * i + 1] for i in range(self.num_layers)]

        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        x = torch.cat([text_embeddings, self.input_linear(sequence)], dim=1)

        out_k, out_v = [], []
        for i, layer in enumerate(self.layers):
            attn_out, nk, nv = self._attention(
                layer.self_attn, layer.norm1(x), k_caches[i], v_caches[i], step
            )
            x = x + attn_out
            x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
            out_k.append(nk)
            out_v.append(nv)

        x = self.out_norm(x)
        conditioning = x[:, -1]  # [B, dim]; ignored on prefill passes
        eos_logit = self.out_eos(conditioning)  # [B, 1]

        new_step = step + x.shape[1]
        result = [conditioning, eos_logit, new_step]
        for i in range(self.num_layers):
            result.append(out_k[i])
            result.append(out_v[i])
        return tuple(result)


class FlowNetWrapper(nn.Module):
    """One stateless LSD step: x_{n+1} = x_n + v(c, s, t, x_n) / N."""

    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net

    def forward(self, c, s, t, x):
        return self.flow_net(c, s, t, x)


class TextConditionerWrapper(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, token_ids):
        from pocket_tts.conditioners.base import TokenizedText

        return self.conditioner(TokenizedText(token_ids))


# --------------------------------------------------------------------------
# Export driver
# --------------------------------------------------------------------------


def export_flow_lm_main(model, out_dir: Path, max_seq: int, opset: int):
    fl = model.flow_lm
    wrapper = FlowLMMainWrapper(fl, max_seq).eval()
    nl, H, Dh = wrapper.num_layers, wrapper.num_heads, wrapper.dim_per_head

    seq = torch.randn(1, 1, fl.ldim)
    text = torch.zeros(1, 0, fl.dim)
    step = torch.zeros(1, dtype=torch.long)
    caches = []
    for _ in range(nl):
        caches.append(torch.zeros(1, max_seq, H, Dh, dtype=torch.float16))
        caches.append(torch.zeros(1, max_seq, H, Dh, dtype=torch.float16))

    in_names = ["sequence", "text_embeddings", "step"]
    out_names = ["conditioning", "eos_logit", "out_step"]
    dyn = {
        "sequence": {0: "batch", 1: "seq_len"},
        "text_embeddings": {0: "batch", 1: "text_len"},
        "step": {0: "batch"},
        "conditioning": {0: "batch"},
        "eos_logit": {0: "batch"},
        "out_step": {0: "batch"},
    }
    for i in range(nl):
        for kv in ("k", "v"):
            in_names.append(f"cache_{kv}_{i}")
            out_names.append(f"out_cache_{kv}_{i}")
            dyn[f"cache_{kv}_{i}"] = {0: "batch"}
            dyn[f"out_cache_{kv}_{i}"] = {0: "batch"}

    path = out_dir / "flow_lm_main.onnx"
    torch.onnx.export(
        wrapper,
        (seq, text, step, *caches),
        str(path),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn,
        opset_version=opset,
        dynamo=False,
    )
    print(f"  flow_lm_main.onnx      {path.stat().st_size / 1e6:6.1f} MB")
    return wrapper


def export_flow_lm_flow(model, out_dir: Path, opset: int):
    fl = model.flow_lm
    wrapper = FlowNetWrapper(fl).eval()
    args = (
        torch.randn(1, fl.dim),
        torch.tensor([[0.0]]),
        torch.tensor([[1.0]]),
        torch.randn(1, fl.ldim),
    )
    path = out_dir / "flow_lm_flow.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={n: {0: "batch"} for n in ("c", "s", "t", "x", "flow_dir")},
        opset_version=opset,
        dynamo=False,
    )
    print(f"  flow_lm_flow.onnx      {path.stat().st_size / 1e6:6.1f} MB")
    return wrapper


def export_text_conditioner(model, out_dir: Path, opset: int):
    wrapper = TextConditionerWrapper(model.flow_lm.conditioner).eval()
    path = out_dir / "text_conditioner.onnx"
    torch.onnx.export(
        wrapper,
        (torch.randint(0, 1000, (1, 20)),),
        str(path),
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={"token_ids": {0: "batch", 1: "seq_len"},
                      "embeddings": {0: "batch", 1: "seq_len"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"  text_conditioner.onnx  {path.stat().st_size / 1e6:6.1f} MB")
    return wrapper



def export_mimi(model, out_dir: Path, cache_len: int, opset: int):
    """Encoder and decoder. Patching must happen after the FlowLM graphs are out."""
    mimi_export.patch_for_onnx()

    enc = mimi_export.MimiEncoderWrapper(model).eval()
    path = out_dir / "mimi_encoder.onnx"
    torch.onnx.export(
        enc,
        (torch.randn(1, 1, mimi_export.ENCODER_TRACE_SAMPLES),),
        str(path),
        input_names=["audio"],
        output_names=["conditioning"],
        dynamic_axes={"audio": {0: "batch", 2: "audio_len"},
                      "conditioning": {0: "batch", 1: "frames"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"  mimi_encoder.onnx      {path.stat().st_size / 1e6:6.1f} MB")

    steps_per_latent = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
    states = mimi_export.decode_states(model.mimi, batch_size=1, cache_len=cache_len)
    names, tensors = mimi_export.flatten(states)
    dec = mimi_export.MimiDecoderWrapper(model, names, steps_per_latent).eval()

    in_names = ["latent"] + [f"state_{i}" for i in range(len(names))]
    out_names = ["audio"] + [f"out_state_{i}" for i in range(len(names))]
    dyn = {"latent": {0: "batch", 1: "frames"}, "audio": {0: "batch", 2: "samples"}}
    for i in range(len(names)):
        dyn[f"state_{i}"] = {0: "batch"}
        dyn[f"out_state_{i}"] = {0: "batch"}

    path = out_dir / "mimi_decoder.onnx"
    torch.onnx.export(
        dec,
        (torch.randn(1, 1, model.flow_lm.ldim), *tensors),
        str(path),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn,
        opset_version=opset,
        dynamo=False,
    )
    print(f"  mimi_decoder.onnx      {path.stat().st_size / 1e6:6.1f} MB"
          f"  ({len(names)} states, steps/latent={steps_per_latent})")

    return [
        {"name": f"state_{i}", "path": n, "dtype": str(t.dtype).removeprefix("torch."),
         "shape": list(t.shape), "fill": "zeros"}
        for i, (n, t) in enumerate(zip(names, tensors))
    ]


def write_manifest(model, out_dir: Path, max_seq: int, wrapper: FlowLMMainWrapper,
                   mimi_state=None, bundle_name: str = "nanotts"):
    fl = model.flow_lm
    cfg = model.config
    manifest = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "sample_rate": cfg.mimi.sample_rate,
        "frame_rate": cfg.mimi.frame_rate,
        "samples_per_frame": int(round(cfg.mimi.sample_rate / cfg.mimi.frame_rate)),
        "latent_dim": fl.ldim,
        "conditioning_dim": fl.dim,
        "num_layers": wrapper.num_layers,
        "num_heads": wrapper.num_heads,
        "dim_per_head": wrapper.dim_per_head,
        "max_seq": max_seq,
        "mimi_cache_len": MIMI_CACHE_LEN,
        "rope_max_period": wrapper.rope_max_period,
        "insert_bos_before_voice": bool(fl.insert_bos_before_voice),
        "vocab_size": int(fl.conditioner.embed.weight.shape[0]) - 1,
        "pad_with_spaces_for_short_inputs": bool(model.pad_with_spaces_for_short_inputs),
        "remove_semicolons": bool(model.remove_semicolons),
        "model_recommended_frames_after_eos": model.model_recommended_frames_after_eos,
        "max_token_per_chunk": 50,
        # -1.0, which is what GenVoice's card quotes for this checkpoint. An
        # earlier version of this file used pocket-tts's own -4.0 because it
        # measured better on WER; that was the wrong instrument. The EOS logit
        # rises unevenly at the end of a phrase and briefly touches -4 before
        # falling back, so the lower threshold stops early: measured on the
        # server across twenty phrases, nineteen of them end sooner and 6.2% of
        # the total duration disappears -- the release of the final syllable,
        # which ASR scores identically (4.62% WER either way) and a listener
        # hears at once.
        "defaults": {"temperature": 0.5, "eos_threshold": -1.0, "lsd_steps": 1},
        "flow_lm_state": (
            [{"name": "step", "dtype": "int64", "shape": ["batch"], "fill": "zeros"}]
            + [
                {
                    "name": f"cache_{kv}_{i}",
                    "dtype": "float16",
                    "shape": ["batch", max_seq, wrapper.num_heads, wrapper.dim_per_head],
                    "fill": "zeros",
                }
                for i in range(wrapper.num_layers)
                for kv in ("k", "v")
            ]
        ),
        "mimi_state": mimi_state or [],
    }
    if fl.insert_bos_before_voice:
        import numpy as np

        np.save(out_dir / "bos_before_voice.npy",
                fl.bos_before_voice.detach().cpu().numpy().astype("float32"))

    (out_dir / "bundle.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  bundle.json            {len(manifest['flow_lm_state'])} flow + "
          f"{len(manifest['mimi_state'])} mimi states")


def main():
    ap = argparse.ArgumentParser(
        description="Export a Pocket TTS checkpoint -- the Kyutai base models or a fine-tune "
                    "such as genvoice/xVibePocketTTS -- to the five ONNX graphs this runtime uses."
    )
    # Either a config (local path, https:// or hf://<repo>/<file>[@rev]) or one of
    # pocket-tts's built-in language names. The two are mutually exclusive upstream.
    ap.add_argument("--config", default=None)
    ap.add_argument("--language", default=None,
                    help="built-in pocket-tts config: english, german, spanish_24l, ...")
    ap.add_argument("--output-dir", default="/out")
    ap.add_argument("--name", default=None, help="bundle_name written into the manifest")
    ap.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--export",
        nargs="+",
        default=["flow_main", "flow", "text", "mimi"],
        choices=["flow_main", "flow", "text", "mimi"],
    )
    args = ap.parse_args()
    if args.config and args.language:
        ap.error("--config and --language are mutually exclusive")
    if not args.config and not args.language:
        args.config = "/work/russian_local.yaml"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.config or args.language} ...")
    model = TTSModel.load_model(config=args.config, language=args.language,
                                temp=0.5, eos_threshold=-4.0)
    model.eval()

    print(f"Exporting (max_seq={args.max_seq}, opset={args.opset}) ...")
    main_wrapper = None
    mimi_state = None
    with torch.no_grad():
        if "flow_main" in args.export:
            main_wrapper = export_flow_lm_main(model, out_dir, args.max_seq, args.opset)
        if "flow" in args.export:
            export_flow_lm_flow(model, out_dir, args.opset)
        if "text" in args.export:
            export_text_conditioner(model, out_dir, args.opset)
        if "mimi" in args.export:
            mimi_state = export_mimi(model, out_dir, MIMI_CACHE_LEN, args.opset)

    if main_wrapper is not None:
        write_manifest(model, out_dir, args.max_seq, main_wrapper, mimi_state,
                       bundle_name=args.name or (args.language or "custom"))

    # The runtime looks for the tokenizer next to the graphs, and pocket-tts may
    # have pulled it from HuggingFace into a cache that does not outlive this
    # container. Serialising it out of the loaded processor sidesteps resolving
    # the config's path a second time.
    tok_out = out_dir / "tokenizer.model"
    proto = model.flow_lm.conditioner.tokenizer.sp.serialized_model_proto()
    tok_out.write_bytes(proto)
    print(f"  tokenizer.model  {len(proto) / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
