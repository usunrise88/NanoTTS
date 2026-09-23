"""ONNX export for the Mimi codec halves.

The upstream streaming modules mutate their state in place, which the tracer
cannot follow, so the state has to be turned into explicit graph I/O. That is
what `patch_for_onnx()` does. It is deliberately called only once the FlowLM
graphs are already exported, because it also rewrites the shared KV-cache
backend that `init_states` uses.

Two deviations from upstream worth knowing about:

  * The Mimi transformer KV cache becomes a circular buffer of `context` slots
    (250) instead of one sized to the whole generation. Attention never looks
    further back than `context`, so nothing is lost and the state stops growing
    with utterance length.
  * Caches are zero-filled rather than NaN-filled. Upstream relies on slicing
    off the unwritten tail; here the mask drops those slots instead, and a
    masked NaN *value* would still poison the weighted sum.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.modules.attention import _LinearKVCacheBackend
from pocket_tts.modules import conv as conv_mod
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.stateful_module import StatefulModule, init_states

# Modules on the decode path. Everything else in MimiModel belongs to the
# encoder and would otherwise ride along as dead pass-through graph I/O.
DECODE_PREFIXES = ("quantizer", "upsample", "decoder")

# 10 s at 24 kHz, and a whole number of 1920-sample frames. Tracing at a
# length that is NOT frame-aligned would bake a non-zero pad into the graph,
# so the runtime contract is: pad reference audio up to a frame multiple.
ENCODER_TRACE_SAMPLES = 1920 * 125

_PATCHED = False


def patch_for_onnx() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def extra_padding(x, kernel_size, stride, padding_total=0):
        # int() pins the value at trace time. That is only sound because the
        # traced length is frame-aligned, which makes the result 0 for every
        # frame-aligned length the graph will ever see.
        import math as _math

        length = int(x.shape[-1])
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal = (_math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return int(ideal - length)

    conv_mod.get_extra_padding_for_conv1d = extra_padding

    def conv1d_forward(self, x, model_state):
        B, C, T = x.shape
        state = self.init_state(B, 0) if model_state is None else self.get_state(model_state)
        TP = state["previous"].shape[-1]

        if TP and self.pad_mode == "replicate":
            new_prev = torch.where(
                state["first"].view(-1, 1, 1),
                x[..., :1].expand_as(state["previous"]),
                state["previous"],
            )
            state["previous"] = new_prev
            if model_state is not None:
                self.get_state(model_state)["previous"] = new_prev

        if TP:
            x = torch.cat([state["previous"], x], dim=-1)
        y = self.conv(x)
        if TP and model_state is not None:
            self.get_state(model_state)["previous"] = x[..., -TP:]
            if self.pad_mode == "replicate":
                self.get_state(model_state)["first"] = torch.zeros_like(state["first"])
        return y

    def convtr_forward(self, x, mimi_state):
        state = self.get_state(mimi_state)
        partial = state["partial"]
        y = self.convtr(x)
        PT = partial.shape[-1]
        if PT > 0:
            head = y[..., :PT] + partial
            new_partial = y[..., -PT:]
            if self.convtr.bias is not None:
                new_partial = new_partial - self.convtr.bias[:, None]
            state["partial"] = new_partial
            y = torch.cat([head, y[..., PT:-PT]], dim=-1)
        return y

    _orig_init_state = _LinearKVCacheBackend.init_state

    def init_state_split(self, batch_size, sequence_length, device, dtype):
        orig = _orig_init_state(self, batch_size, sequence_length, device, dtype)
        zeros = torch.zeros_like(orig["cache"][0], dtype=torch.float16)
        return {
            "cache_k": zeros.clone(),
            "cache_v": zeros.clone(),
            "offset": orig["offset"],
        }

    def append_and_get(self, k, v, state):
        if state is None:
            k_attn, v_attn = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            B = k_attn.shape[0]
            pos_k = torch.arange(k_attn.shape[2], device=k.device, dtype=torch.long)
            pos_k = pos_k.view(1, -1).expand(B, -1)
            return k_attn, v_attn, pos_k, torch.zeros(B, device=k.device, dtype=torch.long)

        cache_k, cache_v = state["cache_k"], state["cache_v"]
        offset = state["offset"]
        B, T_new, H, D = k.shape
        capacity = cache_k.shape[1]

        arange_t = torch.arange(T_new, device=k.device, dtype=torch.long).view(1, T_new)
        write_pos = (offset.view(B, 1) + arange_t) % capacity
        idx = write_pos.view(B, T_new, 1, 1).expand(B, T_new, H, D)
        updated_k = cache_k.scatter(1, idx, k.half())
        updated_v = cache_v.scatter(1, idx, v.half())
        state["cache_k"], state["cache_v"] = updated_k, updated_v

        # Absolute position of every slot, so the existing mask can drop both
        # the not-yet-written slots and anything outside the context window.
        last_abs = (offset.view(B, 1) + T_new) - 1
        slots = torch.arange(capacity, device=k.device, dtype=torch.long).view(1, -1)
        delta = slots - (last_abs % capacity)
        abs_pos = torch.where(delta <= 0, last_abs + delta, last_abs + delta - capacity)
        pos_k = torch.where(abs_pos >= 0, abs_pos, torch.full_like(abs_pos, -1))

        return (
            updated_k.float().permute(0, 2, 1, 3),
            updated_v.float().permute(0, 2, 1, 3),
            pos_k,
            offset,
        )

    def increment_step(self, state, increment):
        state["offset"] = state["offset"] + increment

    StreamingConv1d.forward = conv1d_forward
    StreamingConvTranspose1d.forward = convtr_forward
    _LinearKVCacheBackend.init_state = init_state_split
    _LinearKVCacheBackend.append_and_get = append_and_get
    _LinearKVCacheBackend.increment_step = increment_step


def decode_states(mimi, batch_size: int, cache_len: int):
    full = init_states(mimi, batch_size=batch_size, sequence_length=cache_len)
    return {m: s for m, s in full.items() if m.startswith(DECODE_PREFIXES)}


def flatten(states):
    names, tensors = [], []
    for module in sorted(states):
        for key in sorted(states[module]):
            names.append(f"{module}/{key}")
            tensors.append(states[module][key])
    return names, tensors


class MimiEncoderWrapper(nn.Module):
    """Audio -> conditioning embeddings, speaker projection baked in.

    Stateless: `encode_to_latent` runs every submodule with `model_state=None`.
    """

    def __init__(self, model):
        super().__init__()
        self.mimi = model.mimi
        self.register_buffer("speaker_proj_weight", model.flow_lm.speaker_proj_weight)

    def forward(self, audio):
        # encode_to_latent already returns [B, T, C] in pocket-tts 3.0.2.
        latents = self.mimi.encode_to_latent(audio).to(torch.float32)
        return F.linear(latents, self.speaker_proj_weight)


class MimiDecoderWrapper(nn.Module):
    """Latent frames + state -> audio, with the latent un-normalisation inlined.

    `steps_per_latent` (16 here: 200 Hz encoder rate over 12.5 Hz latent rate) is
    how far the streaming state advances per decoded frame. Upstream decodes one
    frame per call; decoding N at once is equivalent because the transformer is
    causal and the convolutions carry their own overlap state.
    """

    def __init__(self, model, state_names, steps_per_latent: int):
        super().__init__()
        self.mimi = model.mimi
        self.state_names = state_names
        self.steps_per_latent = steps_per_latent
        self.register_buffer("emb_std", model.flow_lm.emb_std)
        self.register_buffer("emb_mean", model.flow_lm.emb_mean)
        self._stateful = {
            name: mod
            for name, mod in model.mimi.named_modules()
            if isinstance(mod, StatefulModule) and name.startswith(DECODE_PREFIXES)
        }

    def forward(self, latent, *flat_state):
        state = {}
        for name, tensor in zip(self.state_names, flat_state):
            module, key = name.rsplit("/", 1)
            state.setdefault(module, {})[key] = tensor

        audio = self.mimi.decode_from_latent(latent * self.emb_std + self.emb_mean, state)

        increment = latent.shape[1] * self.steps_per_latent
        for name, mod in self._stateful.items():
            if name in state:
                mod.increment_step(state[name], increment)

        return (audio, *[state[n.rsplit("/", 1)[0]][n.rsplit("/", 1)[1]] for n in self.state_names])
