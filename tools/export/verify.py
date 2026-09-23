#!/usr/bin/env python3
"""Stage-0 verification for the xVibe ONNX export.

Three questions, in order of how badly a failure would hurt:

  A. Does the hand-rolled transformer layer in FlowLMMainWrapper compute the
     same thing as pocket_tts' own StreamingTransformer? (port fidelity)
  B. Does ONNX Runtime reproduce the torch wrapper? (export fidelity)
  C. Can one batch hold rows sitting at different positions, each matching an
     independent batch-1 run? (this is what continuous batching rests on)
"""

import numpy as np
import onnxruntime as ort
import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

from export_xvibe import FlowLMMainWrapper

MAX_SEQ = 512
torch.manual_seed(0)


def report(name, a, b, tol):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    amax = float(np.abs(a - b).max())
    rms = float(np.sqrt(((a - b) ** 2).mean()))
    ok = amax <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<46} max|d|={amax:.3e}  rms={rms:.3e}")
    return ok


def empty_state(nl, H, Dh, B):
    step = torch.zeros(B, dtype=torch.long)
    caches = []
    for _ in range(nl):
        caches.append(torch.zeros(B, MAX_SEQ, H, Dh, dtype=torch.float16))
        caches.append(torch.zeros(B, MAX_SEQ, H, Dh, dtype=torch.float16))
    return step, caches


def main():
    model = TTSModel.load_model(config="/work/russian_local.yaml", temp=0.5, eos_threshold=-4.0)
    model.eval()
    fl = model.flow_lm
    w = FlowLMMainWrapper(fl, MAX_SEQ).eval()
    nl, H, Dh, dim, ldim = w.num_layers, w.num_heads, w.dim_per_head, fl.dim, fl.ldim
    ok = True

    # ---------------------------------------------------------------- A
    print("\nA. hand-rolled layer vs pocket_tts StreamingTransformer")
    T_pre = 37
    text = torch.randn(1, T_pre, dim)
    seq_ar = torch.randn(1, 1, ldim)

    with torch.no_grad():
        # upstream: prefill then one AR step, through its own stateful path
        up_state = init_states(fl, batch_size=1, sequence_length=MAX_SEQ)
        x = torch.cat([text, fl.input_linear(torch.zeros(1, 0, ldim))], dim=1)
        up_out = fl.transformer(x, up_state)
        increment_steps(fl, up_state, x.shape[1])
        up_pre_cond = fl.out_norm(up_out)[:, -1]

        x2 = torch.cat([torch.zeros(1, 0, dim), fl.input_linear(seq_ar)], dim=1)
        up_out2 = fl.transformer(x2, up_state)
        increment_steps(fl, up_state, x2.shape[1])
        up_ar_cond = fl.out_norm(up_out2)[:, -1]
        up_ar_eos = fl.out_eos(up_ar_cond)

        # ours
        step, caches = empty_state(nl, H, Dh, 1)
        r = w(torch.zeros(1, 0, ldim), text, step, *caches)
        my_pre_cond, step, caches = r[0], r[2], list(r[3:])
        r = w(seq_ar, torch.zeros(1, 0, dim), step, *caches)
        my_ar_cond, my_ar_eos, step, caches = r[0], r[1], r[2], list(r[3:])

    ok &= report("prefill conditioning", up_pre_cond, my_pre_cond, 2e-3)
    ok &= report("AR conditioning", up_ar_cond, my_ar_cond, 2e-3)
    ok &= report("AR eos_logit", up_ar_eos, my_ar_eos, 2e-3)
    print(f"        step after prefill+AR: {step.tolist()} (expected [{T_pre + 1}])")
    ok &= step.tolist() == [T_pre + 1]

    # ---------------------------------------------------------------- B
    print("\nB. ONNX Runtime vs torch wrapper")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession("/out/flow_lm_main.onnx", so, providers=["CPUExecutionProvider"])

    def run_ort(sequence, text_emb, step, caches):
        feed = {
            "sequence": sequence.numpy(),
            "text_embeddings": text_emb.numpy(),
            "step": step.numpy(),
        }
        for i in range(nl):
            feed[f"cache_k_{i}"] = caches[2 * i].numpy()
            feed[f"cache_v_{i}"] = caches[2 * i + 1].numpy()
        out = sess.run(None, feed)
        return out[0], out[1], torch.from_numpy(out[2]), [torch.from_numpy(t) for t in out[3:]]

    step, caches = empty_state(nl, H, Dh, 1)
    o_cond, _, o_step, o_caches = run_ort(torch.zeros(1, 0, ldim), text, step, caches)
    ok &= report("prefill conditioning", up_pre_cond, o_cond, 2e-3)
    o_cond, o_eos, o_step, o_caches = run_ort(seq_ar, torch.zeros(1, 0, dim), o_step, o_caches)
    ok &= report("AR conditioning", up_ar_cond, o_cond, 2e-3)
    ok &= report("AR eos_logit", up_ar_eos, o_eos, 2e-3)

    # several AR steps, to catch drift in the cache round-trip
    cur = seq_ar
    with torch.no_grad():
        t_step, t_caches = step, caches
        t_step, t_caches = empty_state(nl, H, Dh, 1)
        r = w(torch.zeros(1, 0, ldim), text, t_step, *t_caches)
        t_step, t_caches = r[2], list(r[3:])
        s2, c2 = empty_state(nl, H, Dh, 1)
        _, _, s2, c2 = run_ort(torch.zeros(1, 0, ldim), text, s2, c2)
        for n in range(8):
            r = w(cur, torch.zeros(1, 0, dim), t_step, *t_caches)
            t_cond, t_step, t_caches = r[0], r[2], list(r[3:])
            o_cond, _, s2, c2 = run_ort(cur, torch.zeros(1, 0, dim), s2, c2)
            cur = torch.randn(1, 1, ldim)
        ok &= report("conditioning after 8 AR steps", t_cond, o_cond, 3e-3)

    # ---------------------------------------------------------------- C
    print("\nC. heterogeneous batch: rows at different positions")
    Ta, Tb = 41, 17  # deliberately unequal prefixes
    text_a, text_b = torch.randn(1, Ta, dim), torch.randn(1, Tb, dim)
    seq_a, seq_b = torch.randn(1, 1, ldim), torch.randn(1, 1, ldim)

    # independent batch-1 runs
    solo = {}
    for tag, txt, sq in (("a", text_a, seq_a), ("b", text_b, seq_b)):
        s, c = empty_state(nl, H, Dh, 1)
        _, _, s, c = run_ort(torch.zeros(1, 0, ldim), txt, s, c)
        cond, eos, s, c = run_ort(sq, torch.zeros(1, 0, dim), s, c)
        solo[tag] = (cond, eos, s.clone(), [x.clone() for x in c])

    # same two rows packed into one batch, each carrying its own step
    s_a, c_a = solo["a"][2], solo["a"][3]
    s_b, c_b = solo["b"][2], solo["b"][3]
    # rewind one AR step: rebuild post-prefill state, then step both together
    def prefilled(txt):
        s, c = empty_state(nl, H, Dh, 1)
        _, _, s, c = run_ort(torch.zeros(1, 0, ldim), txt, s, c)
        return s, c

    pa_s, pa_c = prefilled(text_a)
    pb_s, pb_c = prefilled(text_b)
    bstep = torch.cat([pa_s, pb_s])
    bcaches = [torch.cat([pa_c[i], pb_c[i]], dim=0) for i in range(2 * nl)]
    bseq = torch.cat([seq_a, seq_b], dim=0)
    bcond, beos, bstep_out, _ = run_ort(bseq, torch.zeros(2, 0, dim), bstep, bcaches)

    print(f"        steps in batch: {bstep.tolist()} -> {bstep_out.tolist()}")
    ok &= report("row a: batched vs solo conditioning", solo["a"][0][0], bcond[0], 3e-3)
    ok &= report("row b: batched vs solo conditioning", solo["b"][0][0], bcond[1], 3e-3)
    ok &= report("row a: batched vs solo eos", solo["a"][1][0], beos[0], 3e-3)
    ok &= report("row b: batched vs solo eos", solo["b"][1][0], beos[1], 3e-3)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
