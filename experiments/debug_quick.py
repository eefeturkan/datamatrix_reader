"""Quick: sample scores at known grid, visualize heatmap, decode with all thresholds."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape
resp = resp_img.astype(np.float32) / 255.0

x0, y0 = 63.0, 18.0
xp = (388.0 - 63.0) / 19.0  # 17.1
yp = (366.0 - 18.0) / 19.0  # 18.3
print(f"Grid: x0={x0} y0={y0} xp={xp:.2f} yp={yp:.2f}")

# Sample scores
pr = 4
scores = np.zeros((20, 20), dtype=np.float32)
for ri in range(20):
    gy = int(round(y0 + ri * yp))
    for ci in range(20):
        gx = int(round(x0 + ci * xp))
        y1 = max(0, gy-pr); y2 = min(h, gy+pr+1)
        x1 = max(0, gx-pr); x2 = min(w, gx+pr+1)
        patch = resp[y1:y2, x1:x2]
        scores[ri, ci] = float(patch.max()) if patch.size else 0.0

print(f"Scores: min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}")
print(f"Row means: {np.round(scores.mean(axis=1), 2)}")
print(f"Col means: {np.round(scores.mean(axis=0), 2)}")

# Heatmap
scores_vis = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
scores_big = cv2.resize(scores_vis, (400, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("debug/cropped_scores_heatmap.png", scores_big)

# Grid overlay on raw
vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
for ri in range(20):
    gy = int(round(y0 + ri * yp))
    for ci in range(20):
        gx = int(round(x0 + ci * xp))
        score = scores[ri, ci]
        intensity = int(score * 255)
        cv2.circle(vis, (gx, gy), 5, (0, intensity, 255 - intensity), -1)
cv2.imwrite("debug/cropped_grid_overlay.png", vis)
print("Saved overlay and heatmap")

# Try all thresholds
best = None
for sq in np.arange(0.05, 0.95, 0.01):
    tv = float(np.quantile(scores, sq))
    bits = (scores >= tv).astype(np.uint8)
    bits[0, :] = 1; bits[:, 0] = 1
    orient_score, oriented = _orient_bits(bits)
    rendered = _render_bits(oriented)
    decoded = _decode_pure_render(rendered)
    if decoded:
        print(f"DECODED sq={sq:.2f} tv={tv:.3f}: {decoded}")
        cv2.imwrite("debug/cropped_score_DECODED.png", rendered)
        best = decoded
        break

if not best:
    print("\nNo decode at any threshold.")
    print("Saving renders at sq=0.3, 0.4, 0.5, 0.6, 0.7:")
    for sq in (0.3, 0.4, 0.5, 0.6, 0.7):
        tv = float(np.quantile(scores, sq))
        bits = (scores >= tv).astype(np.uint8)
        bits[0, :] = 1; bits[:, 0] = 1
        _, oriented = _orient_bits(bits)
        rendered = _render_bits(oriented)
        cv2.imwrite(f"debug/cropped_sq{int(sq*10)}.png", rendered)
        print(f"  sq={sq:.1f}: fill={bits.mean():.2%}")

print("Done!")
