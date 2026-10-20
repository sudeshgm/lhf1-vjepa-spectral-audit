#!/usr/bin/env python3
"""LHF1 Block 2: null calibration of SI_temporal, bootstrapped H3, clean-vs-shuffle linear probes.

Re-encodes clean + shuffle for every clip (shuffle permutation reproduced exactly from Exp -1's
per-clip seed) and keeps ONLY per-clip agreement scores (no dense fields):
  - top-200 PCs (from Exp -1 pca_basis.npz)
  - 200 random orthonormal directions (null a)
  - clean-vs-clean with identity permutation, fresh rng seed (null b), top-50 PCs
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import e_minus_1_temporal_spectrum as E

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--clips", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True, help="Exp -1 output dir"); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--null-seed", type=int, default=12345)
ap.add_argument("--n-random", type=int, default=200); ap.add_argument("--n-pcs", type=int, default=200)
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--n-clips", type=int, default=None); ap.add_argument("--skip-encode", action="store_true")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)

B = np.load(a.exp / "pca_basis.npz"); m1, V = B["mean"], B["V"].astype(np.float64); d = V.shape[0]
R = np.linalg.qr(np.random.default_rng(a.null_seed).standard_normal((d, a.n_random)))[0]  # orthonormal
files = sorted(a.clips.glob("*.npy"))[: a.n_clips]
cache = a.out / "per_clip_agreement.npz"
if not a.skip_encode:
    enc = E.build_encoder(a.checkpoint, "final", np.load(files[0]).shape)
    pcs, rnd, ident = [], [], []; t0 = time.time()
    for ci, f in enumerate(files):
        clip = np.load(f)
        rng = np.random.default_rng(a.seed * 100003 + ci)  # identical to Exp -1 (clean consumes no rng)
        Xc = E.encode(enc, clip); Xs = E.encode(enc, E.apply_intervention(clip, "shuffle", rng))
        rng_b = np.random.default_rng(a.null_seed + ci)
        Xi = E.encode(enc, clip[np.arange(len(clip))])  # identity "shuffle", independent forward
        _ = rng_b.permutation(len(clip))  # seed consumed; permutation deliberately not applied
        P = np.hstack([V[:, : a.n_pcs], R])
        Zc, Zs, Zi = ((X - m1) @ P > 0 for X in (Xc, Xs, Xi))
        ag = (Zc == Zs).mean(0); pcs.append(ag[: a.n_pcs]); rnd.append(ag[a.n_pcs:])
        ident.append((Zc[:, :50] == Zi[:, :50]).mean(0))
        print(f"clip {ci+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    np.savez(cache, pcs=np.array(pcs), rnd=np.array(rnd), ident=np.array(ident))
C = np.load(cache); pcs, rnd, ident = C["pcs"], C["rnd"], C["ident"]; N = len(pcs)

# SI_temporal = mean over clips of sign agreement; "carries order" = LOW SI_temporal.
si_pc, si_rnd = pcs.mean(0), rnd.mean(0)
# Calibrated threshold: 5th percentile of null (a) -> a PC is order-carrying if its SI_temporal is
# below what 95% of random directions reach (one-sided, "lower than null").
# The brief says "95th percentile"; we report both conventions (see caveats).
thr_lo, thr_hi = float(np.percentile(si_rnd, 5)), float(np.percentile(si_rnd, 95))

def frac(si, lo, hi, thr, below=True): return float(((si[lo:hi] < thr) if below else (si[lo:hi] > thr)).mean())
def boot(lo, hi, thr, below=True):
    rng = np.random.default_rng(0); v = []
    for _ in range(a.boot):
        idx = rng.integers(0, N, N); v.append(frac(pcs[idx].mean(0), lo, hi, thr, below))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
h3 = {}
for name, thr, below in [("order_carrying_below_null_p5", thr_lo, True), ("high_SI_above_null_p95", thr_hi, False),
                         ("order_carrying_below_0.7", 0.7, True), ("high_SI_above_0.7", 0.7, False)]:
    f50 = frac(si_pc, 0, 50, thr, below)
    h3[name] = dict(threshold=thr, top50=f50, top50_ci95=boot(0, 50, thr, below),
                    pc51_200=frac(si_pc, 50, 200, thr, below), pc51_200_ci95=boot(50, 200, thr, below), ge_10pct=f50 >= 0.10)

# ---- probes (clip-level splits)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
T = np.load(a.exp / "pooled_tokens.npz"); c, s = T["clean_temporal"].astype(np.float64), T["shuffle_temporal"].astype(np.float64)
Np = len(c); X0 = np.concatenate([c, s]) - m1; y = np.r_[np.zeros(Np), np.ones(Np)]; g = np.r_[np.arange(Np), np.arange(Np)]
def probe(F):
    F = F.reshape(len(F), -1); accs = []
    for tr, te in GroupKFold(5).split(F, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"))
        accs.append(clf.fit(F[tr], y[tr]).score(F[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))
rm = lambda idx: X0 - (X0 @ V[:, idx]) @ V[:, idx].T
probes = {"full": probe(X0), "top50_pcs_only": probe(X0 @ V[:, :50])}
for k in (5, 10, 20, 50):
    probes[f"var_top{k}_removed"] = probe(rm(np.arange(k)))
    r = [probe(rm(np.random.default_rng(sd).choice(d, k, replace=False)))[0] for sd in range(5)]
    probes[f"rand{k}_removed_5seeds"] = (float(np.mean(r)), float(np.std(r)))

# ---- plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.arange(1, len(si_pc) + 1), si_pc, ".-", lw=.8, label="PCs (variance rank)")
ax.axhspan(si_rnd.min(), si_rnd.max(), color="gray", alpha=.2, label="random dirs (min-max)")
ax.axhline(thr_lo, color="r", ls="--", label=f"null p5 = {thr_lo:.3f}")
ax.axhline(thr_hi, color="m", ls=":", label=f"null p95 = {thr_hi:.3f}")
ax.axhline(.7, color="k", ls="-.", lw=.8, label="old 0.7"); ax.axvline(50.5, color="g", lw=.6)
ax.set_xscale("log"); ax.set_xlabel("PC variance rank"); ax.set_ylabel("SI_temporal (clean vs shuffle sign agreement)")
ax.set_title(f"V-JEPA 2 ViT-L final layer, {N} K400-val clips"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(a.out / "si_temporal_vs_rank.png", dpi=150)

S = dict(n_clips=N, n_random=a.n_random, null_a=dict(mean=float(si_rnd.mean()), std=float(si_rnd.std()), p5=thr_lo, p95=thr_hi),
         null_b_identity=dict(mean=float(ident.mean()), min=float(ident.min())),
         si_temporal_top50=dict(mean=float(si_pc[:50].mean()), min=float(si_pc[:50].min()), max=float(si_pc[:50].max())),
         si_temporal_pc51_200_mean=float(si_pc[50:].mean()), H3=h3,
         probe_clean_vs_shuffle_acc_mean_std=probes, si_temporal_pcs=si_pc.tolist(), si_temporal_random=si_rnd.tolist())
(a.out / "summary.json").write_text(json.dumps(S, indent=1))
print(json.dumps({k: v for k, v in S.items() if not k.startswith("si_temporal_") or k.endswith("mean") or k == "si_temporal_top50"}, indent=1))
