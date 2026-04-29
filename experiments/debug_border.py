"""Find EXACT grid by maximizing solid border score, then decode."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape

# morphsub_31: dark=dots present, bright=no dot
# INVERT: noktalar yüksek değer olsun
resp = 1.0 - resp_img.astype(np.float32) / 255.0  

# Find best x0 by maximizing column 0 score (solid border = all dark dots)
# Col 0 should have 20 dark dots - so resp (inverted) should be HIGH at all 20 positions
# Similarly row 0 should be all dark

# Sweep x0 in [30, 100] with various pitches
print("Finding optimal grid by solid border score...")

best_score = -1
best_params = None

# We know approximate pitch from previous: xp~17.1, yp~18.3
for x0 in np.arange(30, 100, 0.5):
    for xp in np.arange(15.5, 19.0, 0.1):
        # Score: sum of resp at col0 positions (20 positions)
        col0_score = 0
        for ri in range(20):
            gx = int(round(x0))  # col 0
            # We need y0 too - estimate as best y0 for this x0
            # For now just use approximate y range
            pass
        
        # Better: sum resp at col0 x for all y in range
        gx0 = int(round(x0))
        if gx0 < 0 or gx0 >= w:
            continue
        col_vals = resp[:, max(0, gx0-3):min(w, gx0+4)].max(axis=1)
        
        # Score: col_vals should peak at regular intervals of yp
        for y0 in np.arange(0, 30, 0.5):
            for yp in np.arange(16.0, 20.0, 0.2):
                ys = [int(round(y0 + ri * yp)) for ri in range(20)]
                ys = [y for y in ys if 0 <= y < h]
                if len(ys) < 10:
                    continue
                score = sum(float(col_vals[y]) for y in ys)
                if score > best_score:
                    best_score = score
                    best_params = (x0, y0, xp, yp)

if best_params:
    x0, y0, xp, yp = best_params
    print(f"Best: x0={x0:.1f} y0={y0:.1f} xp={xp:.1f} yp={yp:.1f} score={best_score:.2f}")
    
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
    
    print(f"Scores: mean={scores.mean():.3f}")
    print(f"Row 0 mean: {scores[0,:].mean():.3f}")
    print(f"Col 0 mean: {scores[:,0].mean():.3f}")
    
    for sq in np.arange(0.30, 0.75, 0.02):
        tv = float(np.quantile(scores, sq))
        bits = (scores >= tv).astype(np.uint8)
        bits[0, :] = 1; bits[:, 0] = 1
        _, oriented = _orient_bits(bits)
        rendered = _render_bits(oriented)
        decoded = _decode_pure_render(rendered)
        if decoded:
            print(f"DECODED sq={sq:.2f}: {decoded}")
            cv2.imwrite("debug/cropped_border_DECODED.png", rendered)

# Alternative: use the raw image directly
# In the original image, DATA MATRIX dots appear as circular depressions
# which look darker in some lighting and brighter in others
# Try the raw image response
print("\n\nTrying raw image response...")
for resp_name, resp_src in [
    ("raw_inv", 1.0 - raw.astype(np.float32)/255.0),  # dark dots -> high
    ("raw", raw.astype(np.float32)/255.0),              # bright centers -> high
]:
    resp = resp_src
    x0, y0 = 63.0, 18.0
    xp = (388.0 - 63.0) / 19.0
    yp = (366.0 - 18.0) / 19.0
    
    pr = 5
    scores = np.zeros((20, 20), dtype=np.float32)
    for ri in range(20):
        gy = int(round(y0 + ri * yp))
        for ci in range(20):
            gx = int(round(x0 + ci * xp))
            y1 = max(0, gy-pr); y2 = min(h, gy+pr+1)
            x1 = max(0, gx-pr); x2 = min(w, gx+pr+1)
            patch = resp[y1:y2, x1:x2]
            scores[ri, ci] = float(patch.max()) if patch.size else 0.0
    
    r0_mean = scores[0,:].mean()
    c0_mean = scores[:,0].mean()
    print(f"  {resp_name}: mean={scores.mean():.3f} row0={r0_mean:.3f} col0={c0_mean:.3f}")
    
    for sq in np.arange(0.30, 0.75, 0.03):
        tv = float(np.quantile(scores, sq))
        bits = (scores >= tv).astype(np.uint8)
        bits[0, :] = 1; bits[:, 0] = 1
        _, oriented = _orient_bits(bits)
        rendered = _render_bits(oriented)
        decoded = _decode_pure_render(rendered)
        if decoded:
            print(f"  DECODED! sq={sq:.2f}: {decoded}")
            cv2.imwrite(f"debug/cropped_raw_{resp_name}_DECODED.png", rendered)

print("Done!")
