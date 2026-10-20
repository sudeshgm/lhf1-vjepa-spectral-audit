# LHF-1 — Does V-JEPA put time in its noise or in its signal?

**Status:** not started · **Priority:** best fit for existing work · **Budget:** 2–4 weeks, frozen checkpoints

> Plugs directly into `../../vjepa_temporal/`. Shares tooling with
> `../p3_temporal_embedding_diagnostics/`.

---

## The claim in one paragraph

*Suppressing Non-Semantic Noise in Masked Image Modeling Representations* (Hjelkrem-Tan et al.,
CVPR 2026) shows that masked-image-modelling encoders bury **non-semantic positional noise in
their strongest principal components** — a pathology "pervasive in MIM-based models while nearly
non-existent in other SSL models" (they compare MAE, iBOT, DINOv2, DINOv3, CAPI, I-JEPA, DINO,
DeiT3). Their fix, SOAP, is a post-hoc Gram-Schmidt projection removing components with high
Semantic Invariance, and it improves zero-shot kNN classification and TokenCut segmentation
across every MIM model tested. Their own limitations section concedes the tension: **positional
information IS useful for dense prediction.** Video is the decisive test case, because temporal
position is sometimes noise and sometimes the entire signal. Absolute frame index is irrelevant
to "is this a kitchen"; it is the whole answer to "did the car cut in before or after the brake".
This project measures where video MIM encoders — VideoMAE, V-JEPA, V-JEPA 2, VideoMAEv2 — put
time in their spectrum, and shows that the fix everyone is about to copy from images destroys
exactly the tasks video exists for.

## Why it is the right project for you

- It reuses the V-JEPA infrastructure, checkpoints and driving clips you already have in
  `../../vjepa_temporal/`.
- It is pure linear algebra on frozen features. No training run, no pretraining, no annotation.
- Both outcomes are publishable: either video MIM has the same pathology (and the fix has a
  trade-off nobody has characterised) or it does not (and MIM-video is structurally different
  from MIM-image, which is a finding about the objective).
- It composes with your existing factorization note: SOAP's spectral decomposition and the
  appearance/ego/agent split are the same kind of operator on the same token space.

## Experiment −1 — the two-day go/no-go

Take one V-JEPA checkpoint you already have loaded. PCA the patch-token embeddings over ~200
driving clips. For each of the top 50 components, compute a **temporal Semantic Invariance
score**: how much does the component's activation change when you (a) shuffle the frame order,
(b) freeze the clip to a repeated single frame?

**Stop rule.** If temporal information lives entirely in the low-variance tail of the spectrum,
SOAP-style suppression of the top components is harmless for temporal tasks, there is no
trade-off to report, and the paper has no tension. Stop.

## Budget

| Phase | Compute |
|---|---|
| E−1 spectral scan, 1 model | ~4 GPU-hours |
| Full sweep, 4–5 video encoders × 3 layers | ~30–50 GPU-hours |
| Downstream trade-off curves | ~20 GPU-hours |

Dump pooled and low-rank summaries, not dense token fields. Dense fp16 tokens are ~60 MB per
clip per layer at ViT-G scale — the same storage trap flagged in the `vjepa_temporal` risk
register.

---

See [PLAN.md](PLAN.md) for the full note.

---
## Repo quickstart (Exp −1)
- Encoder: V-JEPA 2 ViT-L `facebook/vjepa2-vitl-fpc64-256` (HF, MIT, 326M params, 1.30 GB safetensors), loaded with `transformers.AutoModel`; final-layer tokens `[B, 2048, 1024]` for a 16×256×256 clip (8×16×16 grid).
- Clips: 200 Kinetics-400 val stand-ins (one per class), `scripts/prepare_clips.py`; manifest in `results/e_minus_1/clips_manifest.json`. The tagged **driving clips are not available** yet.
- `python scripts/e_minus_1_temporal_spectrum.py --checkpoint facebook/vjepa2-vitl-fpc64-256 --clips <dir> --out results/e_minus_1/vjepa2_vitl_final`
- Tests: `pytest -q tests` (random-weight tiny ViT, CPU). SOAP details: `NOTES_soap.md`.
