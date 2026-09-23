**English** · [Русский](README.ru.md)

# NanoTTS

A production CPU inference server for small speech models:
[Kyutai **Pocket TTS**](https://huggingface.co/kyutai/pocket-tts) and its
fine-tunes, and the **Supertonic 3** family (Supertonic 3 itself and
TeraTTS v2). Derived from
[VolgaGerm/PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) (MIT) and
rewritten for multi-threaded serving, NUMA locality and an OpenAI-compatible
HTTP API.

These models are small enough to be genuinely fast on a CPU, and this is what it
takes to serve them: warmed voices that start from a `memcpy`, stateless ONNX
graphs, weights replicated per NUMA node, and streaming from the first frame.

The two architectures have almost nothing in common below the waterline — one is
autoregressive over a 12.5 Hz codec, the other sizes an utterance and solves it
in a single flow-matching pass — so they sit behind a backend interface. The
HTTP API, streaming, the audio formats, NUMA placement, metrics and the console
are shared; `architecture` in the bundle manifest picks the rest.

```bash
curl -fsSL https://raw.githubusercontent.com/usunrise88/NanoTTS/main/install.sh | bash
```

## Install

The one-liner fetches the source, exports the checkpoint to ONNX, builds the
runtime image and starts it on `127.0.0.1:8080`. Everything runs in Docker; the
only things it puts on the host are one directory and a `nanotts` launcher.

```bash
# a Pocket TTS base language instead of the default Russian fine-tune
curl -fsSL .../install.sh | bash -s -- --model english

# expose it, pick a port, stay in fp32
curl -fsSL .../install.sh | bash -s -- --bind 0.0.0.0 --port 9000 --precision fp32

curl -fsSL .../install.sh | bash -s -- update
curl -fsSL .../install.sh | bash -s -- uninstall           # keeps warmed voices
curl -fsSL .../install.sh | bash -s -- uninstall --purge    # removes them too
```

Afterwards: `nanotts start | stop | restart | status | logs | update | uninstall`.

Requirements: Linux on x86_64 (AVX2 strongly preferred), Docker with Compose v2,
git, and about 12 GiB of disk for the export.

## Models

### Pocket TTS and its fine-tunes (`architecture: pocket`)

Anything the `pocket-tts` package can load, because the exporter goes through
`TTSModel.load_model`:

| `--model` | What it is |
|---|---|
| `english`, `english_2026-04`, `english_2026-04_24l` | Kyutai base, English |
| `french_24l`, `german`, `german_24l`, `italian`, `italian_24l` | Kyutai base |
| `portuguese`, `portuguese_24l`, `spanish`, `spanish_24l` | Kyutai base |
| `genvoice/xVibePocketTTS` | Russian fine-tune — the default |
| `owner/name` | any HuggingFace repo holding a `config.yaml` |
| `hf://…`, `https://…`, `/path/to/config.yaml` | passed straight through |

The runtime reads its geometry from the exported manifest — layer count, heads,
latent dim, frame rate — so the 24-layer variants work exactly like the 6-layer
ones.

### Supertonic 3 and TeraTTS v2 (`architecture: s3`)

| `--model` | What it is |
|---|---|
| `TeraSpace/TeraTTSv2` | Russian and English, 44.1 kHz, ten shipped voices |
| `Supertone/supertonic-3` | English, Korean and Japanese, 44.1 kHz |

These two release ONNX rather than weights, so there is nothing to trace: the
install downloads the graphs and converts the voices. They share an
architecture — the graph interfaces are identical down to the tensor shapes,
only one output name differs — but not their text conventions, and those live
in the manifest rather than in code.

They differ from Pocket TTS in two ways that matter more than the file format:

- **TTFB grows with the length of the text.** A duration predictor sizes the
  whole utterance and one flow-matching pass fills all of it in before anything
  can be heard; only the vocoder streams. Pocket TTS emits its first frame
  immediately and does not care how long the text is. Measured on the reference
  machine: 564 ms for a three-word sentence, 755 ms for a ten-word one, against
  a flat ~171 ms without stress marks for Pocket.
- **No voice cloning.** The releases ship fixed style vectors and no style
  encoder, so `POST /v1/voices` answers 501 and says why. You get the voices
  the checkpoint came with.

TeraTTS also has a 134-character vocabulary with no digits, so numbers are
spelled out before they reach the encoder — unexpanded, a digit is not
mispronounced, it is dropped along with what it meant. The speller matches
num2words on 382 of 382 reference cases, including Russian gender agreement.

**Everything measured below was measured on the Russian Pocket fine-tune.** The
other checkpoints go through the same code, but their numbers are not ours to
quote.

## What's in it

- **HTTP server** with `/v1/audio/speech`, chunked streaming, `wav` / `pcm` / `mp3`.
- **Web console** for trying it: synthesis, uploading and warming your own voices
  into `.safetensors`, live monitoring. Compiled into the binary.
- **NUMA placement**: one engine per node, weights replicated, threads pinned,
  memory bound locally.
- **Warmed voices** as saved FlowLM KV caches. A request starts with a `memcpy`
  instead of a prefill.
- **Stress marks** through a RUAccent sidecar for Russian, where the checkpoint
  was trained on text carrying U+0301 and synthesising without it costs 17.7% WER
  against 3.2%.
- **Export** into five ONNX graphs with a dynamic batch axis and per-row positions.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/audio/speech` | synthesis; `input`, `voice`, `response_format`, plus a `nanotts` object for the extra knobs |
| GET | `/v1/models`, `/v1/voices` | catalogues |
| POST | `/v1/voices` | multipart `id` + `file` (WAV) → warm-up → `.safetensors` |
| GET | `/v1/voices/{id}/state` | download the warmed state |
| DELETE | `/v1/voices/{id}` | remove a voice |
| GET | `/healthz`, `/readyz`, `/stats`, `/metrics` | operations |

```bash
curl -X POST localhost:8080/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input": "Ст+арый з+амок сто+ит д+орого, а дв+ерь закрыв+ает зам+ок.",
  "voice": "male_deep",
  "response_format": "wav",
  "nanotts": {"temperature": 0.5, "eos_threshold": -4.0, "seed": 42}
}' --output out.wav
```

The extras sit in their own object so an OpenAI client that knows nothing about
them keeps working. `xvibe` is accepted there as an alias for `nanotts`, because
that is what the object was called before the project was renamed.

## EOS threshold

The default is **−4.0**, which is `pocket-tts`'s own. GenVoice's model card
quotes `EOS −1` as a condition of their benchmark, and taking that as a serving
default proved expensive: at −1.0 the model regularly reads the text and then
keeps going, inventing several seconds of speech. Measured on matched seeds:

| set | −1.0 | −4.0 |
|---|---|---|
| short phrases (20) | WER 5.59% / CER 1.60% | 6.21% / 1.70% |
| long phrases (8) | 5.65% / 2.14% | **4.84% / 2.03%** |
| long paragraph, 6 seeds | 16.20% / 9.68% | **9.72% / 2.94%** |

The regression on the short set is one utterance out of twenty — four of its five
errors are identical under both thresholds, and none is a truncated ending. The
threefold CER drop on the paragraph is exactly the invented tail going away.

Beyond the threshold, a chunk's generation is capped the way upstream does it, at
`ceil((tokens/3 + 2) × 12.5)` frames. Without that, a missed EOS ran to
`max_frames` — forty seconds of whatever it felt like.

## Stress marks (Russian)

The checkpoint was trained on text carrying U+0301 and falls apart without it.
Measured on matched seeds, transcribed through ASR:

| set | no stress | RUAccent | hand-written |
|---|---|---|---|
| long phrases (8) | WER 17.74% | **3.23%** | 3.23% |
| ordinary set (20) | WER 20.50% | **6.21%** | 9.32% |

RUAccent beats hand-written marks too. It lives in its own sidecar (`accent/`)
because it is Python with no native port; the models are baked into the image, so
it starts without network access. Wire it up with `--accent-url`; without it the
server places no stress and expects the caller to.

The price is latency: TTFB 171 → ~300 ms p50 on ordinary prose, up to ~650 ms on
text dense with homographs. The distribution is bimodal — an ordinary sentence is
marked in 30–60 ms, a sentence containing a homograph costs about 600 ms because
the resolver model runs. Softened by a text cache, by four ONNX threads per
instance (median 102 → 59 ms), and by giving the sidecar its own NUMA node,
without which it cost another 180 ms.

Only as much text as the first chunk needs is marked on the critical path. The
cut is **on a sentence boundary** and chunking happens after marking, so the
chunk boundaries are the ones the unsplit text would have produced. The reverse
order — chunk first, then mark the pieces — is cheaper and wrong: the token
budget would have to be guessed down to leave room for the stress marks, which
breaks long sentences at commas the model would rather have read through.

Hand-written stress always wins: `+` before a vowel, or U+0301 after one. Those
words are swapped for placeholders before the sidecar sees them and put back
afterwards. Redundant marks on `ё` are dropped, and `+` stays literal where no
Russian vowel follows, so `C++` and `2+2` survive. Per request:
`"nanotts": {"auto_accent": false}`.

## NUMA

The reference machine is an EPYC 7351P: one socket, but **four NUMA nodes** of
four cores and 32 GB, and eight CCXs with 8 MB of L3 each. A remote access costs
16 against 10 local, so weights are replicated per node — an int8 bundle is
~110 MB, four copies is 440 MB out of 128 GB — rather than shared.

```
--nodes 0,1,2,3        which nodes to occupy (default: all)
--workers-per-node N   several engines per node; cores are divided between them
--threads N            intra-op threads per engine (default: physical cores)
--smt                  use SMT siblings as well
--no-pin               no pinning or memory binding (containers without CAP_SYS_NICE)
```

Order matters: the creating thread pins itself and switches its memory policy to
`MPOL_BIND` first, and only then are the ORT sessions built — otherwise
first-touch puts the weights on the wrong node. ORT's own threads are pinned
through `session.intra_op_thread_affinities`.

`nanotts topology` prints the layout it detected.

## Numbers

EPYC 7351P, one worker per node, int8, 20 phrases, `schedutil` governor:

| Concurrency | Throughput | TTFB p50 | TTFB p95 | RTF p50 |
|---|---|---|---|---|
| 1 | 4.03× realtime | 171 ms | 192 ms | 0.251 |
| 4 | 9.02× realtime | 268 ms | 374 ms | 0.423 |
| 8 | 8.94× realtime | 2029 ms | 2723 ms | 0.848 |

Saturation arrives at a concurrency equal to the worker count; past that,
requests queue. Lifting that ceiling is what continuous batching is for — the
graphs for it are exported already (below), the scheduler is not written.

```bash
export ASR_KEY=...
python3 tools/bench/bench.py --url http://localhost:8080 --voice male_deep --seed 1000 \
  --concurrency 4 --asr-url http://asr-host/v1 --asr-key "$ASR_KEY"
```

## How the export is put together

Five graphs, all stateless — state travels through inputs and outputs:

| Graph | Role |
|---|---|
| `flow_lm_main` | AR backbone, split K/V cache in fp16, EOS head |
| `flow_lm_flow` | one step of the LSD solver: `(c, s, t, x) → flow_dir` |
| `text_conditioner` | tokens → embeddings |
| `mimi_encoder` | audio → 1024-dim conditioning (`speaker_proj` folded in) |
| `mimi_decoder` | latents + state → audio |

Two departures from PocketTTS.cpp's export:

- **`step` is a `[B]` vector rather than a shared scalar**, and RoPE takes a
  per-row offset. Rows are independent and a slot may sit at any position. That is
  what makes real continuous batching possible instead of lockstep cohorts, and it
  removes the need for upstream's `pad` alignment. Verified: two rows at positions
  41 and 17 in one batch give **bit-identical** results to independent runs.
- **One shared `step` instead of per-layer ones**: 13 state tensors rather than 18.

Contract: reference audio for the encoder must be a multiple of 1920 samples — the
graph is traced frame-aligned, otherwise non-zero padding is baked into it. The
runtime pads to that itself.

## Cloning references

The server measures an uploaded WAV and keeps it next to the voice along with a
report: duration, peak, speech level, DC, clipping, silence fraction, crest
factor. Warm-up always "succeeds" — the encoder is equally happy to encode speech
and noise — so without that measurement a bad clone leaves no evidence.

Measured on 20 phrases through ASR: GenVoice's shipped state 9.32% WER, a warm-up
from the same audio 8.70%, a live 44.1 kHz recording 13.04%. Duration itself
barely matters (4.4 s of clean audio gives 10.56%), silence at the edges is
harmless, and a low level costs about six points.

## Limits

- **Batching is not implemented**, although the graphs support it and that is
  verified. The ceiling is the worker count.
- **The cost of int8 is not resolved by WER.** int8 takes RTF from 0.35 to 0.25;
  what it costs in quality drowns in noise on these corpora. Three seed bases, WER:

  | | short (20) | long (8) |
  |---|---|---|
  | int8 per-tensor | 6.21 · 6.21 · 8.07 → 6.83% | 4.84 · 4.03 · 2.42 → 3.76% |
  | int8 per-channel | 6.83 · 11.18 · 7.45 → 8.49% | 1.61 · 5.65 · 0.81 → 2.69% |
  | fp32 (one seed) | 7.45% | 4.03% |

  The seed-to-seed spread on one corpus reaches five points — larger than any
  difference between variants — and the direction flips between corpora. An earlier
  note here claiming "int8 costs 1.9 points of WER" rested on exactly one such pair
  and does not reproduce. Answering the question needs a corpus the size of
  GenVoice's 800 generations, not 28 phrases.

  More to the point: **WER is the wrong instrument.** ASR transcribes cleanly
  through timbre damage, so it cannot see what a listener hears immediately.
  Comparing generated audio against fp32 does not work either — the loop is
  autoregressive, a difference in the first frame sends the runs down different
  trajectories, and what comes out is two distinct realisations of the sentence
  rather than a reference and a copy (SNR −1.8 dB, i.e. uncorrelated).

  Measure the graphs directly instead: 600 frames of teacher forcing, every variant
  driven by the same fp32 latent, outputs compared.

  | | FlowLM backbone | LSD head | EOS logit |
  |---|---|---|---|
  | int8 per-tensor | 9.58% | 5.52% | 0.103 |
  | **int8 per-channel** | **8.44%** | **3.84%** | 0.103 |

  Hence per-channel as the default: closer to fp32 everywhere, and 30% closer on
  the LSD head, which writes the fine structure of each frame — the part heard as
  timbre. Same file size (76.2 vs 75.9 MB), same RTF, and the EOS logit does not
  move.

- **fp16 and bf16 are pointless on AVX2.** Zen 1 has F16C, which converts between
  fp16 and fp32 but does no arithmetic, so ORT casts up to fp32 and computes there.
  On the GEMV that dominates the AR step, 1×1024 @ 1024×1024: fp32 48.6 µs, fp16
  57.7 µs. Half precision as a middle ground between int8 and fp32 does not exist
  on this hardware.
- **The weights' license is undetermined.** `genvoice/xVibePocketTTS` has no
  license field on HuggingFace; it states only that the Kyutai base is declared
  CC BY 4.0. For production, settle the terms with GenVoice. The code here and in
  every dependency is MIT/Apache — see [NOTICE](NOTICE).
- The `schedutil` governor does clock loaded cores up to 2.9 GHz, but with a lag
  that shows up in TTFB. Use `performance` in production.

## Layout

```
src/model/     bundle.json, states, tokenizer
src/engine/    ORT sessions, AR loop, LSD solver, decoder pipeline
src/text/      stress, normalisation, token chunking
src/voice/     warm-up, registry, safetensors
src/audio/     wav, resampler (Kaiser polyphase), mp3
src/numa/      topology, pinning, local allocation
src/http/      server, OpenAI API, metrics
web/           console (Vite + TanStack Router, compiled into the binary)
accent/        RUAccent sidecar
tools/export/  ONNX export, quantisation, Python reference runtime
tools/bench/   load and WER/CER through ASR
```

`tools/export/pyrt.py` is a reference implementation of the same pipeline in
Python. It exists so the C++ has something to be checked against: the text layer
is already token-for-token identical to it.

## License

MIT — see [LICENSE](LICENSE). Attribution, and the situation with the model
weights, are in [NOTICE](NOTICE).
