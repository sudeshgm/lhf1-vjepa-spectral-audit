# NOTES: how SOAP (dsb-ifi/soap) computes SI and the projector

Source read: `soap/soap.py`, `soap/welford.py`, `fit_projector.py`, `notebook.ipynb` (pip-installed from git+https://github.com/dsb-ifi/soap.git).

## 1. PCA basis
`WelfordChanEstimator` streams patch tokens X [n, d] (Welford/Chan merge of mean m1 and scatter m2),
cov = m2/(n-correction). `eigh(cov)` -> eigvals ascending, V. "Rank" order = descending variance.
PCA response of a token x on component k: z_k = (x - m1) · v_k  (`weights = V`, `bias = -m1 @ V`).

## 2. Aggregated patch responses (p)
For each dataset (real images, and synthetic images from `SynthesizedDataSet`: pink noise, DCT,
gradients, procedural), over all images:
`p[i, j, k] = mean_over_images( 1[ z_k(token at grid pos (i,j)) > 0 ] )`  (binary=True default),
shape [H', W', d], components in rank (descending-variance) order. Soft version uses raw z.

## 3. SI score
Per position and component, with p = p_data, q = p_synth clamped to [eps, 1-eps], alpha=2, gamma(=`alpha` arg of score fn)=1:

    SI(p,q) = 2 (p q + (1-p)(1-q))^gamma / ( (p^a + (1-p)^a)^(1/a) + (q^a + (1-q)^a)^(1/a) + eps ),  a = 2

`semantic_invariance_score = SI(p_data, p_synth).mean(dims=[0,1]).flip(-1)`  -> one score per component,
flipped from rank order back to `eigh` ascending order. High SI = component responds identically on
real and semantics-free synthetic data = positional / non-semantic artifact.

## 4. Score transform (mu, tau)
`semantic_invariance_score_transform`: sort scores descending (rank r = 0,1,...) and multiply by

    w_r = sigmoid( (mu - 0.5 - r) / tau )

(soft top-mu selection; mu, tau are softplus-parametrised in `ScoreTransform`). Defaults: function
mu=1.5, tau=0.2 (`SOAP.compute_from_data` defaults mu=2.0, tau=0.05; `from_precomputed` -> mu=2.0, tau=0.05).
If mu=None, mu = #(scores > threshold=0.75). Then un-sort to original order.

## 5. Projector
    P = I - V W V^T,   W = diag(transformed scores),  V = eigh eigenvectors (ascending order, matching flip)
Applied as `x @ P^T`. Hard variant (`manual_truncation`): P = I - V_S V_S^T for chosen components S.
Note P is applied to raw (uncentred) tokens.

## Video version in this repo (scripts/e_minus_1_temporal_spectrum.py)
- Tokens: V-JEPA 2 grid T'xH'xW' = 8x16x16 (tubelet 2, patch 16) for 16x256x256 clips; PCA over all clean tokens.
- `SI_semantic`: exact SOAP formula on aggregated binary responses p[t,h,w,k], real=clean clips,
  synth=uniform-noise clips (SOAP's synthesizer is image-only; swap-in possible). Scaled with (mu=1.5, tau=0.2).
- `SI_temporal/SI_motion/SI_spatial`: per-clip sign agreement of z_k between clean and
  shuffle/freeze/patchshuf, averaged over clips and positions (= SOAP SI with p,q in {0,1}, per clip).
  Per-clip is needed because aggregated maps of content components are trivially shuffle-invariant.
- Projector baselines use P = I - V W V^T with V in descending order (scores aligned to it).
