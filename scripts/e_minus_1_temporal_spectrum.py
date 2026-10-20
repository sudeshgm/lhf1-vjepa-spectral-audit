#!/usr/bin/env python3
"""Experiment -1: where in the spectrum does a video MIM encoder keep time?

Two-day go/no-go for lhf1_vjepa_spectral_audit. See ../PLAN.md section 5.1 and ../NOTES_soap.md.

Pipeline (streaming; dense token fields are NEVER written to disk):
  pass 1  clean clips -> Welford mean/covariance of patch tokens -> PCA basis V (descending variance)
  pass 2  per clip, all 5 interventions {clean, shuffle, freeze, patchshuf, noise} with a fixed seed:
          - project tokens onto V, binarise (z > 0) exactly like SOAP's aggregated responses
          - per-clip sign agreement vs clean          -> SI_temporal (shuffle), SI_motion (freeze),
                                                          SI_spatial (patchshuf), SI_noise_clip (noise)
          - aggregated per-position responses p[t,h,w,k] (mean over clips) for clean and noise
            -> SI_semantic = SOAP semantic_invariance_score(p_clean, p_noise)   (faithful to SOAP)
          - save pooled tokens only: per-clip spatially pooled [T', d] and global mean [d]
  analysis
          - H3 check: fraction of top-50 PCs whose SI_temporal < thr (i.e. carry temporal ORDER).
            < 10% -> NO-GO (STOP).
          - projector comparison on pooled tokens: SOAP-style SI removal, top-k variance removal
            (no SI), random rank-k removal (>=5 seeds). Metric: temporal-order signal retained
            (||P(f_clean - f_shuffle)|| / ||f_clean - f_shuffle||) and clean-vs-shuffle linear-probe acc.

Outputs (in --out dir): pca_basis.npz, pooled_tokens.npz, si_scores.json, baselines.json, summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INTERVENTIONS = ("clean", "shuffle", "freeze", "patchshuf", "noise")
SI_NAME = {"shuffle": "SI_temporal", "freeze": "SI_motion", "patchshuf": "SI_spatial", "noise": "SI_noise_clip"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- interventions
def apply_intervention(clip: np.ndarray, kind: str, rng: np.random.Generator, patch: int = 16) -> np.ndarray:
    """clip: [T, H, W, C] uint8 -> same shape/dtype. Deterministic given rng state."""
    T, H, W, C = clip.shape
    if kind == "clean":
        return clip
    if kind == "shuffle":
        perm = rng.permutation(T)
        while T > 1 and np.array_equal(perm, np.arange(T)):
            perm = rng.permutation(T)
        return clip[perm]
    if kind == "freeze":
        return np.repeat(clip[T // 2][None], T, axis=0)
    if kind == "noise":
        return rng.integers(0, 256, size=clip.shape, dtype=np.uint8)
    if kind == "patchshuf":
        assert H % patch == 0 and W % patch == 0
        gh, gw = H // patch, W // patch
        tiles = clip.reshape(T, gh, patch, gw, patch, C).transpose(0, 1, 3, 2, 4, 5).reshape(T, gh * gw, patch, patch, C)
        tiles = tiles[:, rng.permutation(gh * gw)]  # same permutation for every frame
        return np.ascontiguousarray(tiles.reshape(T, gh, gw, patch, patch, C).transpose(0, 1, 3, 2, 4, 5).reshape(T, H, W, C))
    raise ValueError(kind)


def to_tensor(clips: np.ndarray) -> torch.Tensor:
    """[B,T,H,W,C] uint8 -> [B,T,C,H,W] float normalised (V-JEPA 2 / HF layout)."""
    x = (clips.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 1, 4, 2, 3).contiguous()


# --------------------------------------------------------------------------- encoders
class TinyVideoViT(nn.Module):
    """Random-weight stand-in with V-JEPA-like tokenisation (tubelet 2 x 16 x 16). Tests only."""

    def __init__(self, dim=64, depth=2, heads=4, tubelet=2, patch=16, frames=16, size=64):
        super().__init__()
        self.tubelet, self.patch = tubelet, patch
        self.grid = (frames // tubelet, size // patch, size // patch)
        self.embed = nn.Conv3d(3, dim, (tubelet, patch, patch), (tubelet, patch, patch))
        self.pos = nn.Parameter(torch.randn(1, int(np.prod(self.grid)), dim) * 0.5)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.embed_dim = dim

    def forward(self, x):  # x [B,T,C,H,W]
        x = self.embed(x.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        return self.blocks(x + self.pos)


class VJEPA2Encoder(nn.Module):
    """HF transformers VJEPA2Model, encoder only (predictor skipped), final-layer patch tokens."""

    def __init__(self, path: str, layer: str = "final"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(path).eval()
        cfg = self.m.config
        self.embed_dim = cfg.hidden_size
        self.layer = layer
        self.tubelet, self.patch = cfg.tubelet_size, cfg.patch_size

    def set_grid(self, T, H, W):
        self.grid = (T // self.tubelet, H // self.patch, W // self.patch)

    def forward(self, x):
        if self.layer == "final":
            return self.m.get_vision_features(x)  # [B, T'*H'*W', d], layernormed encoder output
        out = self.m(x, skip_predictor=True, output_hidden_states=True)
        return out.encoder_hidden_states[int(self.layer)]


def build_encoder(spec: str, layer: str, clip_shape):
    T, H, W, _ = clip_shape
    if spec.startswith("tiny"):
        torch.manual_seed(0)
        return TinyVideoViT(frames=T, size=H).eval()
    enc = VJEPA2Encoder(spec, layer)
    enc.set_grid(T, H, W)
    return enc


# --------------------------------------------------------------------------- SOAP SI (verbatim formula)
def soap_semantic_invariance(p, q, alpha=2.0, gamma=1.0, eps=1e-12):
    """Same as soap.soap.semantic_invariance (see NOTES_soap.md)."""
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    num = (p * q + (1 - p) * (1 - q)) ** gamma
    denp = (p ** alpha + (1 - p) ** alpha) ** (1 / alpha)
    denq = (q ** alpha + (1 - q) ** alpha) ** (1 / alpha)
    return np.clip(2 * num / (denp + denq + eps), eps, 1 - eps)


def soap_score_transform(scores, mu=1.5, tau=0.2):
    """SOAP ScoreTransform: sort descending, multiply rank r by sigmoid((mu-0.5-r)/tau)."""
    idx = np.argsort(-scores)
    r = np.arange(len(scores))
    w = 1 / (1 + np.exp(np.clip(-(mu - 0.5 - r) / tau, -700, 700)))
    out = np.empty_like(scores); out[idx] = scores[idx] * w
    return out


# --------------------------------------------------------------------------- data
def load_clips(clip_dir: Path, limit: int | None):
    files = sorted(clip_dir.glob("*.npy"))[:limit]
    if not files:
        raise FileNotFoundError(f"no .npy clips in {clip_dir}")
    return files


@torch.no_grad()
def encode(enc, clip: np.ndarray) -> np.ndarray:
    return enc(to_tensor(clip[None]))[0].double().numpy()  # [N, d]


# --------------------------------------------------------------------------- main passes
def run(args) -> dict:
    torch.set_num_threads(args.threads)
    files = load_clips(args.clips, args.n_clips)
    first = np.load(files[0])
    enc = build_encoder(args.checkpoint, args.layer, first.shape)
    Tg, Hg, Wg = enc.grid
    d = enc.embed_dim
    K = min(args.n_components, d)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # pass 1: covariance of clean tokens (Welford/Chan, float64)
    n, m1, m2 = 0, np.zeros(d), np.zeros((d, d))
    for f in files:
        X = encode(enc, np.load(f))
        mc = X.mean(0); nc = len(X); M2c = (X - mc).T @ (X - mc)
        delta = mc - m1; nj = n + nc
        m1 = m1 + delta * nc / nj; m2 = m2 + M2c + np.outer(delta, delta) * n * nc / nj; n = nj
    cov = m2 / (n - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, V = evals[order], evecs[:, order]  # descending variance (SOAP "rank" order)
    np.savez(args.out / "pca_basis.npz", mean=m1, eigvals=evals, V=V.astype(np.float32), n_tokens=n)
    t_pass1 = time.time() - t0

    # pass 2: interventions
    agree = {k: np.zeros(d) for k in INTERVENTIONS if k != "clean"}
    agg = {k: np.zeros((Tg * Hg * Wg, d)) for k in INTERVENTIONS}
    pooled_t = {k: [] for k in INTERVENTIONS}
    for ci, f in enumerate(files):
        clip = np.load(f)
        rng = np.random.default_rng(args.seed * 100003 + ci)  # fixed per-clip seed
        bins = {}
        for kind in INTERVENTIONS:
            X = encode(enc, apply_intervention(clip, kind, rng))
            Z = (X - m1) @ V  # PCA responses, all components
            b = (Z > 0).astype(np.float64)
            bins[kind] = b
            agg[kind] += b
            pooled_t[kind].append(X.reshape(Tg, Hg * Wg, d).mean(1).astype(np.float32))
        for kind in agree:
            agree[kind] += (bins[kind] == bins["clean"]).mean(0)
        if args.verbose:
            print(f"clip {ci+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    N = len(files)
    t_total = time.time() - t0

    si = {SI_NAME[k]: (agree[k] / N) for k in agree}
    p_clean, p_noise = agg["clean"] / N, agg["noise"] / N
    si["SI_semantic"] = soap_semantic_invariance(p_clean, p_noise).mean(0)  # SOAP-faithful
    si["SI_semantic_scaled"] = soap_score_transform(si["SI_semantic"], args.mu, args.tau)

    pooled = {k: np.stack(v) for k, v in pooled_t.items()}  # [N, T', d]
    np.savez(args.out / "pooled_tokens.npz", **{f"{k}_temporal": v for k, v in pooled.items()},
             **{f"{k}_global": v.mean(1) for k, v in pooled.items()}, files=np.array([f.name for f in files]))
    (args.out / "si_scores.json").write_text(json.dumps({k: v[:K].tolist() for k, v in si.items()}, indent=1))

    h3 = h3_check(si["SI_temporal"], args.h3_top, args.order_thr, args.h3_min_frac)
    base = baselines(pooled, V, si, ks=args.ks, seeds=args.random_seeds, mu=args.mu, tau=args.tau)
    (args.out / "baselines.json").write_text(json.dumps(base, indent=1))
    summary = dict(checkpoint=args.checkpoint, layer=args.layer, n_clips=N, grid=[Tg, Hg, Wg], dim=d,
                   seed=args.seed, sec_pass1=t_pass1, sec_total=t_total,
                   sec_per_forward=t_total / (N * (1 + len(INTERVENTIONS))), H3=h3,
                   mean_si_top50={k: float(v[:50].mean()) for k, v in si.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["H3"], indent=1))
    return summary


def h3_check(si_temporal, top=50, thr=0.7, min_frac=0.10) -> dict:
    """Among the top-`top` variance PCs, fraction that carry temporal order (SI_temporal < thr)."""
    s = np.asarray(si_temporal)[:top]
    frac = float((s < thr).mean())
    verdict = "GO" if frac >= min_frac else "NO-GO: time lives in the tail, no trade-off to report -> STOP"
    print(f"[H3] top-{top} PCs carrying temporal order (SI_temporal<{thr}): {frac:.1%}  -> {verdict}")
    return dict(top=top, thr=thr, min_frac=min_frac, frac_order=frac,
                n_order=int((s < thr).sum()), go=frac >= min_frac, verdict=verdict)


# --------------------------------------------------------------------------- baselines
def projector(V, idx):
    Vk = V[:, idx]
    return np.eye(V.shape[0]) - Vk @ Vk.T


def soap_projector(V, scores):
    """P = I - V diag(w) V^T  (SOAP, with V in descending-variance order and w scaled SI)."""
    return np.eye(V.shape[0]) - (V * scores) @ V.T


def order_metrics(P, pooled, mean):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    c = (pooled["clean"] - mean) @ P.T; s = (pooled["shuffle"] - mean) @ P.T  # [N,T',d]
    c0 = pooled["clean"] - mean; s0 = pooled["shuffle"] - mean
    retained = float(np.linalg.norm(c - s) / (np.linalg.norm(c0 - s0) + 1e-12))
    X = np.concatenate([c.reshape(len(c), -1), s.reshape(len(s), -1)]); y = np.r_[np.zeros(len(c)), np.ones(len(s))]
    cv = min(5, len(c))
    acc = float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv).mean()) if cv >= 2 else float("nan")
    # appearance proxy: clean-vs-freeze pooled similarity retained (global)
    return dict(order_signal_retained=retained, probe_clean_vs_shuffle_acc=acc)


def baselines(pooled, V, si, ks=(1, 2, 5, 10, 20, 50), seeds=5, mu=1.5, tau=0.2) -> dict:
    assert seeds >= 5, "random rank-k baseline needs >= 5 seeds"
    V = V.astype(np.float64)
    mean = pooled["clean"].reshape(-1, V.shape[0]).mean(0)
    d = V.shape[0]
    out = {"none": order_metrics(np.eye(d), pooled, mean)}
    out["soap_SI_semantic"] = order_metrics(soap_projector(V, si["SI_semantic_scaled"]), pooled, mean)
    si_rank = np.argsort(-si["SI_semantic"])
    for k in ks:
        if k > d:
            continue
        out[f"si_top{k}"] = order_metrics(projector(V, si_rank[:k]), pooled, mean)
        out[f"var_top{k}"] = order_metrics(projector(V, np.arange(k)), pooled, mean)  # no SI
        runs = [order_metrics(projector(V, np.random.default_rng(s).choice(d, k, replace=False)), pooled, mean)
                for s in range(seeds)]
        out[f"rand{k}"] = {m: dict(mean=float(np.mean([r[m] for r in runs])), std=float(np.std([r[m] for r in runs])),
                                   seeds=seeds) for m in runs[0]}
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF dir/id of V-JEPA 2 (e.g. facebook/vjepa2-vitl-fpc64-256) or 'tiny'")
    ap.add_argument("--clips", type=Path, required=True, help="dir of [16,256,256,3] uint8 .npy clips")
    ap.add_argument("--n-clips", type=int, default=None)
    ap.add_argument("--layer", default="final")
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h3-top", type=int, default=50)
    ap.add_argument("--order-thr", type=float, default=0.7)
    ap.add_argument("--h3-min-frac", type=float, default=0.10)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--mu", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/e_minus_1"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
