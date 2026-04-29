"""Polarity fix: noktalar karanlık = high score -> invert resp."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape

# Try both polarities of the response map
for resp_name, resp_raw in [
    ("normal", resp_img.astype(np.float32) / 255.0),
    ("inverted", 1.0 - resp_img.astype(np.float32) / 255.0),
    # Also try morphbright_31 (which highlights the specular highlights)
    ("morphbright", cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphbright_31.png",
                                cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0),
    ("morphbright_inv", 1.0 - cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphbright_31.png",
                                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0),
]:
    resp = resp_raw
    x0, y0 = 63.0, 18.0
    xp = (388.0 - 63.0) / 19.0
    yp = (366.0 - 18.0) / 19.0
    
    # Also try shifting x0/y0 slightly
    for dx0 in (0, -2, 2, -4, 4):
        for dy0 in (0, -2, 2, -4, 4):
            pr = 4
            scores = np.zeros((20, 20), dtype=np.float32)
            for ri in range(20):
                gy = int(round(y0 + dy0 + ri * yp))
                for ci in range(20):
                    gx = int(round(x0 + dx0 + ci * xp))
                    y1 = max(0, gy-pr); y2 = min(h, gy+pr+1)
                    x1 = max(0, gx-pr); x2 = min(w, gx+pr+1)
                    patch = resp[y1:y2, x1:x2]
                    scores[ri, ci] = float(patch.max()) if patch.size else 0.0
            
            for sq in np.arange(0.35, 0.70, 0.03):
                tv = float(np.quantile(scores, sq))
                bits = (scores >= tv).astype(np.uint8)
                bits[0, :] = 1; bits[:, 0] = 1
                orient_score, oriented = _orient_bits(bits)
                rendered = _render_bits(oriented)
                decoded = _decode_pure_render(rendered)
                if decoded:
                    print(f"DECODED! resp={resp_name} dx0={dx0} dy0={dy0} sq={sq:.2f}: {decoded}")
                    cv2.imwrite("debug/cropped_polarity_DECODED.png", rendered)
                    sys.exit(0)

# Also: try morphbright directly
print("No decode with polarity variants. Trying morphbright directly...")
mb = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphbright_31.png", cv2.IMREAD_GRAYSCALE)
print(f"morphbright stats: min={mb.min()} max={mb.max()} mean={mb.mean():.0f}")

# What does morphbright look like at the grid positions?
resp = mb.astype(np.float32) / 255.0
x0, y0 = 63.0, 18.0
xp = (388.0 - 63.0) / 19.0
yp = (366.0 - 18.0) / 19.0
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

print(f"morphbright scores: min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}")
print(f"Row 0 mean (should be solid): {scores[0,:].mean():.3f}")
print(f"Col 0 mean (should be solid): {scores[:,0].mean():.3f}")
print(f"Row 0: {np.round(scores[0,:], 2)}")
print(f"Col 0: {np.round(scores[:,0], 2)}")

# Heatmap
sh = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("debug/cropped_morphbright_heatmap.png", cv2.resize(sh, (400, 400), interpolation=cv2.INTER_NEAREST))
print("Done!")
