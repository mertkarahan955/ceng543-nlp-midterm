#!/usr/bin/env python3
import numpy as np, os, glob, csv

attn_dirs = ["outputs/bahdanau", "outputs/luong", "outputs/scaleddot"]
out_rows = []

def entropy_and_sharpness(attn):
    eps = 1e-12
    attn = attn / (attn.sum(axis=1, keepdims=True) + eps)
    ent = -np.sum(attn * np.log(attn + eps), axis=1)
    sharp = attn.max(axis=1)
    return ent.mean(), sharp.mean()

for d in attn_dirs:
    npz_files = glob.glob(os.path.join(d, "*.npz")) + glob.glob(os.path.join(d, "*.npy"))
    all_ents, all_sharps = [], []
    for f in npz_files:
        data = np.load(f)
        a = data["attn"] if isinstance(data, np.lib.npyio.NpzFile) else data
        if a.ndim == 3:  # (B, T, S) or (T, S)
            a = a.reshape(-1, a.shape[-1])
        mean_ent, mean_sharp = entropy_and_sharpness(a)
        all_ents.append(mean_ent)
        all_sharps.append(mean_sharp)
    if all_ents:
        out_rows.append((os.path.basename(d), np.mean(all_ents), np.mean(all_sharps)))
    else:
        out_rows.append((os.path.basename(d), None, None))

with open("attn_stats_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["attention", "mean_entropy", "mean_sharpness"])
    for r in out_rows:
        w.writerow(r)

print("xsSaved attention statistics to attn_stats_summary.csv")
for r in out_rows:
    name = r[0]
    ent = f"{r[1]:.4f}" if r[1] is not None else "N/A"
    shp = f"{r[2]:.4f}" if r[2] is not None else "N/A"
    print(f"{name:12s} | entropy={ent} | sharpness={shp}")

