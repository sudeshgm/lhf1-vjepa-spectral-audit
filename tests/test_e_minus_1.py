import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e_minus_1_temporal_spectrum as E


def _clip(seed=0, T=16, S=64):
    return np.random.default_rng(seed).integers(0, 256, (T, S, S, 3), dtype=np.uint8)


def test_interventions_shapes_and_determinism():
    c = _clip()
    for k in E.INTERVENTIONS:
        a = E.apply_intervention(c, k, np.random.default_rng(1))
        b = E.apply_intervention(c, k, np.random.default_rng(1))
        assert a.shape == c.shape and a.dtype == np.uint8 and np.array_equal(a, b)


def test_intervention_semantics():
    c = _clip(); r = np.random.default_rng(0)
    s = E.apply_intervention(c, "shuffle", r)
    assert not np.array_equal(s, c) and sorted(map(bytes, s)) == sorted(map(bytes, c))
    f = E.apply_intervention(c, "freeze", r); assert all(np.array_equal(f[i], c[8]) for i in range(16))
    p = E.apply_intervention(c, "patchshuf", r)
    assert not np.array_equal(p, c) and np.array_equal(np.sort(p.reshape(-1)), np.sort(c.reshape(-1)))
    # same spatial permutation across frames: frame-0 tile multiset preserved
    t = lambda x: sorted(x[0].reshape(4, 16, 4, 16, 3).transpose(0, 2, 1, 3, 4).reshape(16, -1).tobytes()[i*768:(i+1)*768] for i in range(16))
    assert t(p) == t(c)


def test_soap_formula_matches_package():
    import torch
    from soap.soap import semantic_invariance
    p, q = np.random.rand(5, 5, 7), np.random.rand(5, 5, 7)
    np.testing.assert_allclose(E.soap_semantic_invariance(p, q), semantic_invariance(torch.tensor(p), torch.tensor(q)).numpy(), rtol=1e-6)


def test_score_transform_matches_package():
    import torch
    from soap.soap import ScoreTransform
    s = np.random.rand(20)
    t = ScoreTransform(mu=1.5, tau=0.2).double()
    srt = torch.sort(torch.tensor(s), descending=True)
    ref = np.empty(20); ref[srt.indices.numpy()] = t(srt.values).detach().numpy()
    np.testing.assert_allclose(E.soap_score_transform(s, 1.5, 0.2), ref, rtol=1e-4, atol=1e-9)


def test_h3_check():
    assert E.h3_check(np.r_[np.full(10, .5), np.full(40, .9)])["go"]
    assert not E.h3_check(np.r_[np.full(4, .5), np.full(46, .9)])["go"]


def test_end_to_end_tiny_vit(tmp_path):
    cd = tmp_path / "clips"; cd.mkdir()
    for i in range(6):
        np.save(cd / f"c{i:03d}.npy", _clip(i))
    out = tmp_path / "out"
    s = E.run(E.parse(["--checkpoint", "tiny", "--clips", str(cd), "--out", str(out), "--ks", "1", "5", "--threads", "2"]))
    assert s["grid"] == [8, 4, 4] and s["dim"] == 64
    files = sorted(p.name for p in out.iterdir())
    assert files == ["baselines.json", "pca_basis.npz", "pooled_tokens.npz", "si_scores.json", "summary.json"]
    pt = np.load(out / "pooled_tokens.npz")
    assert pt["clean_temporal"].shape == (6, 8, 64)  # pooled only, no dense [N,128,64] field
    assert all(pt[k].ndim <= 3 for k in pt.files)
    b = json.loads((out / "baselines.json").read_text())
    for k in ["none", "soap_SI_semantic", "var_top1", "var_top5", "rand1", "rand5", "si_top5"]:
        assert k in b
    assert b["rand5"]["order_signal_retained"]["seeds"] == 5
    si = json.loads((out / "si_scores.json").read_text())
    assert set(si) >= {"SI_temporal", "SI_spatial", "SI_semantic", "SI_motion"}
    assert abs(b["none"]["order_signal_retained"] - 1) < 1e-5
