# WORKING RESEARCH NOTE — LHF-1

## Where does a video MIM encoder keep time?

*A spectral audit of VideoMAE / V-JEPA / V-JEPA 2 representations, and the trade-off nobody has
characterised when the image-domain fix is copied to video.*

Frozen encoders. No pretraining. Linear algebra and linear probes only. The project dies at
Experiment −1 if temporal information does not live in the components the image-domain fix
removes.

---

## 1. The claim

Hjelkrem-Tan et al. (CVPR 2026) establish, for images:

- MIM objectives bias representations toward **positional, non-semantic** structure.
- This structure concentrates in the **strong principal components** of the patch-embedding
  covariance — "strong principal components exhibit high levels of non-semantic noise".
- It is **pervasive in MIM models and nearly absent in other SSL models** (contrast: MAE, iBOT,
  CAPI, I-JEPA vs DINO, DINOv2, DINOv3, DeiT3).
- It is removable post-hoc: a Semantic-Invariance score identifies the offending components, a
  Gram-Schmidt projection removes them, and zero-shot kNN classification and TokenCut salient
  segmentation both improve.
- Their stated limitation: **positional information is useful for dense prediction tasks.** They
  suppress it anyway, and they only test ViTs on images.

The claim of this note is that video is where this stops being a free lunch, and that the
distinction between *spatial* position (usually noise) and *temporal* position (often the entire
signal) has never been measured in a video MIM encoder.

Three claims:

- **C1.** Video MIM encoders (VideoMAE, V-JEPA, V-JEPA 2, VideoMAEv2) exhibit the same
  concentration of positional non-semantic structure in high-variance components as their image
  counterparts, and the effect is stronger for *spatial* than *temporal* position.
- **C2.** Temporal-order information is **not** cleanly separable from the components SOAP
  removes. Suppressing them improves appearance-driven tasks (action recognition by scene, clip
  retrieval) and degrades order-driven tasks (temporal ordering, temporal grounding, dense
  propagation).
- **C3.** A *split* projection — suppress spatial-position components, preserve temporal-position
  components — dominates both the unmodified representation and naive SOAP on the joint
  frontier.

C2 is the lead result and the tension the paper is built on. C3 is the method, and it is three
lines of linear algebra.

## 2. Objects of study

**Encoders (frozen, public):** V-JEPA, V-JEPA 2, VideoMAE, VideoMAEv2, and — as the
non-MIM control that Hjelkrem-Tan's own result predicts should be clean — a video DINO-style or
contrastive encoder (InternVideo2 / ViCLIP). Include an image DINOv2 with per-frame pooling as
the second control.

The non-MIM control is not optional. Hjelkrem-Tan's finding is specifically that the pathology
is *MIM-specific*. If your video contrastive model shows it too, the story changes from "MIM
buries position" to "video encoders bury position", which is a different and arguably better
paper — but you must know which one you are writing.

**Layers:** three depths (early / mid / final). Pilot on one backbone before fixing the set; the
`vjepa_temporal` risk register is right that five backbones × five layers × dense dumps does not
fit on a T4.

**Clips:** driving clips you already have (with maneuver tags), plus a public control set —
Kinetics-400 subset (appearance-dominant), Something-Something v2 (order-dominant, since its
classes are explicitly direction- and order-sensitive).

Something-Something v2 is the single most important dataset here: it is the only widely-used
benchmark where shuffling frames genuinely destroys the label. It is your order-driven task.

## 3. The measurement

### 3.1 Temporal Semantic Invariance

Hjelkrem-Tan score a component by how much it activates on semantic vs non-semantic input. The
temporal version needs three interventions, not one:

| Intervention | What it destroys | What a component that survives is encoding |
|---|---|---|
| **Frame shuffle** | temporal order, preserves frame content and count | order-invariant content |
| **Static freeze** (repeat middle frame) | all motion, preserves appearance | pure appearance |
| **Spatial patch shuffle** | spatial layout, preserves per-patch content | layout-invariant content |
| **Synthetic noise clip** | everything (Hjelkrem-Tan's original control) | nothing; pure positional artifact |

For each principal component `i`, define three scores: `SI_temporal(i)`, `SI_spatial(i)`,
`SI_semantic(i)`. Every component gets a coordinate in that 3-space. **The lead figure of the
paper is the spectrum coloured by that coordinate** — variance rank on the x-axis, and where
each component sits on the noise/appearance/order simplex.

### 3.2 The split projection (C3)

Partition components into `{spatial-position noise, temporal-position signal, semantics}` by
thresholding the three SI scores, then build two Gram-Schmidt projections: `P_spatial_removed`
and `P_naive_SOAP` (removes all high-SI components regardless of type). Compare both against
identity on every downstream task.

## 4. Hypotheses and pre-registered thresholds

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | Video MIM encoders concentrate positional non-semantic structure in high-variance components. | The top-20 components' mean SI score is no higher than the tail's. |
| H2 | The pathology is MIM-specific, as in images. | The video contrastive control shows an equal or larger effect. (Not fatal — it changes the title.) |
| H3 | Temporal-order information overlaps the components naive SOAP removes. | < 10 % of removed components have high `SI_temporal`. **This is the Experiment −1 kill criterion.** |
| H4 | Naive SOAP helps appearance tasks and hurts order tasks. | Something-Something v2 linear-probe accuracy does not drop by ≥ 2 points under naive SOAP. |
| H5 | The split projection dominates both baselines. | Split projection is not Pareto-superior to identity and naive SOAP on the (appearance, order) plane. |

## 5. Experiments, kill-first

### 5.1 Experiment −1 — two days

One V-JEPA checkpoint already loaded, ~200 driving clips, PCA over patch tokens, compute
`SI_temporal` for the top 50 components.

**Stop rule.** If temporal information lives in the low-variance tail (H3 falsified), naive SOAP
is harmless for temporal tasks, there is no trade-off, and the paper has no tension. Stop.

### 5.2 Experiment 0 — controls, one week

1. **Non-MIM control** (InternVideo2 / ViCLIP). Decides H2 and therefore the title.
2. **Image-DINOv2 per-frame control.** If the video spectrum looks identical to per-frame image
   features stacked, the encoder is not doing anything temporal and the whole analysis is about
   an image model.
3. **Reproduce Hjelkrem-Tan on images.** Run their protocol on one image MIM model and check you
   recover their reported improvement. If you cannot reproduce the image result, you cannot
   interpret the video result. Budget two days for this and do not skip it.
4. **Label purity for the driving clips.** The `vjepa_temporal` note already flags that maneuver
   tags are automatic with unmeasured precision. Hand-verify 100 before any number derived from
   them enters the paper, and report the precision.

### 5.3 Experiment 1 — the spectrum figure

Full three-intervention SI scan, 4–5 encoders × 3 layers × 3 clip families. Produce the coloured
spectrum plot. This is the paper's identity.

### 5.4 Experiment 2 — the trade-off curve (C2, the lead result)

Sweep the number of suppressed components from 0 to ~64 and plot two curves on one axis:

- appearance-driven: Kinetics-400 linear probe, clip-level retrieval on your driving set,
- order-driven: Something-Something v2 linear probe, temporal-ordering probe (predict whether
  clip B follows clip A), DAVIS label propagation.

The expected figure is a scissors: one curve rises, the other falls. If both rise, SOAP is free
in video too and the paper becomes a much less interesting "SOAP also works for video" note —
decide by end of week 2.

### 5.5 Experiment 3 — the split projection (C3)

Build `P_spatial_removed`, evaluate on the same axes, show Pareto dominance. Report the
threshold sensitivity honestly; Hjelkrem-Tan's own Fermi-window parameters μ, τ require tuning
and they say so.

### 5.6 Negative controls

- **Random projection of equal rank.** Removing *any* k directions changes performance. The
  comparison is against a random rank-k removal, not against identity. Hjelkrem-Tan's own result
  is only meaningful relative to this control; make sure yours is too.
- **Top-k variance removal without SI scoring.** If simply dropping the top-k components (no
  semantic invariance measurement at all) matches SOAP, the SI machinery is decoration. This is
  the cheapest way for the paper to die and should be run in week 1.

The top-k control is the real risk. In `../p1_scenario_embedding/` and in the `vjepa_temporal`
note the same trivial baseline — "common-component removal" — is flagged as a project-killer.
It is the same baseline. Run it early.

## 6. Metrics and statistics

- Linear probe accuracy (frozen features, single linear layer, fixed schedule across all
  conditions — no per-condition tuning).
- Temporal-ordering probe AUC.
- kNN retrieval Recall@1 on driving clips with maneuver tags.
- DAVIS J&F for dense propagation.
- Effective feature rank (SVD entropy) before and after projection, to connect to Zhang's
  *Residual Connections Harm Generative Representation Learning*, which correlates effective rank
  with representation quality.
- Report every number against the **random rank-k removal** control, not against identity.

## 7. Compute and risk register

| Risk | Reality | Mitigation |
|---|---|---|
| Storage | Dense fp16 tokens ≈ 60 MB per clip per layer at ViT-G. | Dump PCA bases and low-rank summaries, not dense fields. Keep dense tokens only for the ~60 control clips. |
| T4 memory | ViT-G at 64 frames × 384² is ~18 K tokens per clip. | Reduce frames and resolution; the spectrum shape is the claim, not absolute accuracy. Report the config. |
| Checkpoint licensing | The `vjepa_temporal` register already flags that the locally vendored V-JEPA 2.1 ViT-G weights may not be publicly usable. | Confirm license **before writing**. If not public, make a public checkpoint the primary object of study and report 2.1 as a supplementary row. |
| Reproducing the image result | If you cannot reproduce Hjelkrem-Tan on images, nothing downstream is interpretable. | Exp 0 control 3, two days, mandatory. |
| Trivial baseline wins | Top-k variance removal matches SI-guided removal. | Week 1, not week 4. If it wins, publish the one-line finding. |

## 8. Failure modes, in order of likelihood

1. **Top-k variance removal matches SI-guided removal.** → The SI machinery adds nothing.
   Publish the one-liner, abandon the method.
2. **H3 falsified at Exp −1** — time lives in the tail. → No trade-off, no paper. Stop in 2 days.
3. **Both curves rise.** → "SOAP transfers to video" — a modest note, not a CVPR paper.
4. **The non-MIM control shows the same pathology.** → Not fatal, but the framing shifts from
   "MIM buries position" to "video encoders bury position". Rewrite the intro, keep the results.
5. **Driving-clip maneuver tags are too noisy to support the retrieval axis.** → Drop to public
   benchmarks only (Kinetics + SSv2 + DAVIS), which is sufficient.

## 9. Week plan

| Week | Work | Stop rule |
|---|---|---|
| 0 (2 days) | Exp −1: SI_temporal over top-50 PCs, one V-JEPA checkpoint, 200 clips. | Time lives in the tail → stop. |
| 1 | Exp 0: non-MIM control, image-DINOv2 control, reproduce Hjelkrem-Tan on images, tag verification, **top-k trivial baseline**. | Trivial baseline wins → publish one-liner and stop. |
| 2 | Exp 1 spectrum figure across encoders and layers. Exp 2 trade-off curves begin. | Both curves rise → downgrade scope. |
| 3 | Exp 2 complete; Exp 3 split projection; random rank-k controls. | — |
| 4 | Threshold sensitivity, effective-rank analysis, writing. | — |

## 10. Title directions

- *Time Lives in the Components You Were About to Delete*
- *Spatial Noise, Temporal Signal: Auditing the Spectrum of Video MIM Representations*
- *SOAP Does Not Transfer: Positional Structure in Video Masked Modelling*

## 11. Explicit non-goals

- Pretraining a better video encoder.
- Beating state of the art on Something-Something v2.
- Explaining why MIM produces positional structure. Hjelkrem-Tan did not, and the mechanism is
  not needed for the trade-off claim.
