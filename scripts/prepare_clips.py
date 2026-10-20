#!/usr/bin/env python3
"""Decode public stand-in clips -> [16,256,256,3] uint8 .npy (outside git).
Kinetics-400 val (CVDF mirror, part_0.tar.gz). 16 frames uniformly from the middle 2 s
(stride ~ 4 at 30 fps), short-side resize to 256 + centre crop. Picks <=1 clip per class first."""
import argparse, csv, json
from pathlib import Path
import av, numpy as np
from PIL import Image

def decode(path, T=16, S=256):
    with av.open(str(path)) as c:
        frames = [f.to_image() for f in c.decode(video=0)]
    if len(frames) < T:
        return None
    mid = len(frames) // 2; span = min(len(frames), 64)
    idx = np.linspace(mid - span // 2, mid + span // 2 - 1, T).round().astype(int).clip(0, len(frames) - 1)
    out = []
    for i in idx:
        im = frames[i]; w, h = im.size; s = S / min(w, h)
        im = im.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - S) // 2, (h - S) // 2
        out.append(np.asarray(im.crop((l, t, l + S, t + S)).convert("RGB")))
    return np.stack(out).astype(np.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True); ap.add_argument("--csv", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
lab = {r["youtube_id"]: r["label"] for r in csv.DictReader(open(a.csv))}
vids = sorted(a.src.rglob("*.mp4"))
by = {}
for v in vids:
    by.setdefault(lab.get(v.name[:11], "?"), []).append(v)
order = [l[0] for l in by.values()] + [x for l in by.values() for x in l[1:]]
meta = []
for v in order:
    if len(meta) >= a.n: break
    try: x = decode(v)
    except Exception: x = None
    if x is None: continue
    name = f"k400val_{len(meta):03d}.npy"; np.save(a.out / name, x)
    meta.append(dict(file=name, source=v.name, label=lab.get(v.name[:11], "?")))
json.dump(meta, open(a.out / "manifest.json", "w"), indent=1)
print(len(meta), "clips;", len({m["label"] for m in meta}), "classes")
