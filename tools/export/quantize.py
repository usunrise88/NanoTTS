"""Dynamic INT8 quantisation, applied only where the traffic is.

flow_lm_main and flow_lm_flow are read in full on every frame, so shrinking
them is what moves RTF. mimi_decoder is amortised over a whole chunk and its
SEANet convolutions are the part most likely to produce audible artefacts, so
it stays fp32 unless explicitly asked for.

Per-channel, not per-tensor. One scale for a whole weight matrix is set by its
widest output channel, and transformer projections reliably have a few outliers,
which leaves the rest of the columns using a fraction of the 256 levels. Driving
both variants with the same fp32 latent for 600 frames and comparing against the
unquantised graphs: the backbone's output is 9.58% off per-tensor against 8.44%
per-channel, and the LSD head -- which writes the fine structure of each frame,
so the part a listener hears as timbre -- is 5.52% off against 3.84%. The EOS
logit moves by 0.103 either way, so stopping behaviour is unchanged.

None of this is visible in WER: ASR transcribes cleanly through timbre damage,
which is why the audio metric said the two were equivalent and the ear did not.
"""
import argparse, sys
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/out")
ap.add_argument("--models", nargs="+", default=["flow_lm_main", "flow_lm_flow"])
args = ap.parse_args()

for name in args.models:
    src = Path(args.dir) / f"{name}.onnx"
    dst = Path(args.dir) / f"{name}_int8.onnx"
    if not src.exists():
        print(f"  skip {name}: not found"); continue
    tmp = Path(args.dir) / f"{name}.shapes.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(src))), str(tmp))
    try:
        quantize_dynamic(model_input=str(tmp), model_output=str(dst),
                         weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                         per_channel=True,
                         extra_options={"ForceQuantizeNoType": True, "DefaultTensorType": 1})
        a, b = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
        print(f"  {name}: {a:.1f} MB -> {b:.1f} MB ({b / a * 100:.0f}%)")
    finally:
        tmp.unlink(missing_ok=True)
