"""Use response map intensity at each grid position to decide bit value (soft decision)."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render
from pylibdmtx.pylibdmtx import decode as dmtx_decode
import zxingcpp

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape
resp = resp_img.astype(np.float32) / 255.0

# Grid parameters (from best_pts analysis)
# Top-left ~(63, 20), pitch ~17px
# BUT we need to refine. Let's sweep more carefully.

# From previous: 175 points with thresh=95, pitch=16.8
# Point bounds: x=[63,388] y=[18,366]
# 19 intervals in x: (388-63)/19 = 17.1
# 19 intervals in y: (366-18)/19 = 18.3

# Let's try rectangle pitch (non-square grid)
x0_base = 63.0
y0_base = 18.0
xp_base = (388.0 - 63.0) / 19.0  # 17.1
yp_base = (366.0 - 18.0) / 19.0  # 18.3

print(f"Grid params: x0={x0_base:.1f} y0={y0_base:.1f} xp={xp_base:.1f} yp={yp_base:.1f}")

best_decoded = None
best_score = -1

for dx0 in np.arange(-5, 5.1, 0.5):
    for dy0 in np.arange(-5, 5.1, 0.5):
        for dxp in np.arange(-1.5, 1.6, 0.2):
            for dyp in np.arange(-1.5, 1.6, 0.2):
                x0 = x0_base + dx0
                y0 = y0_base + dy0
                xp = xp_base + dxp
                yp = yp_base + dyp
                
                # Sample response at each grid point
                scores = np.zeros((20, 20), dtype=np.float32)
                for ri in range(20):
                    gy = y0 + ri * yp
                    if gy < 0 or gy >= h: break
                    for ci in range(20):
                        gx = x0 + ci * xp
                        if gx < 0 or gx >= w: break
                        # Sample a 5x5 window max
                        pr = 4
                        y1 = max(0, int(gy)-pr); y2 = min(h, int(gy)+pr+1)
                        x1 = max(0, int(gx)-pr); x2 = min(w, int(gx)+pr+1)
                        patch = resp[y1:y2, x1:x2]
                        scores[ri, ci] = float(patch.max()) if patch.size else 0.0
                
                # Try multiple thresholds
                for sq in np.arange(0.25, 0.65, 0.05):
                    tv = float(np.quantile(scores, sq))
                    bits = (scores >= tv).astype(np.uint8)
                    bits[0, :] = 1; bits[:, 0] = 1
                    
                    orient_score, oriented = _orient_bits(bits)
                    rendered = _render_bits(oriented)
                    decoded = _decode_pure_render(rendered)
                    
                    if decoded:
                        score = orient_score
                        if score > best_score:
                            best_score = score
                            best_decoded = decoded
                            print(f"DECODED! dx0={dx0:.1f} dy0={dy0:.1f} dxp={dxp:.1f} dyp={dyp:.1f} sq={sq:.2f}: {decoded}")
                            cv2.imwrite("debug/cropped_FINAL_SUCCESS.png", rendered)
                            
                            # Grid visualization
                            vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                            for ri in range(20):
                                for ci in range(20):
                                    gx = int(x0 + ci * xp)
                                    gy = int(y0 + ri * yp)
                                    color = (0, 0, 255) if bits[ri, ci] else (100, 200, 100)
                                    cv2.circle(vis, (gx, gy), 4, color, -1)
                            cv2.imwrite("debug/cropped_FINAL_grid.png", vis)

if not best_decoded:
    print("\nStill no decode. Trying with lower threshold (accept more false positives)...")
    
    x0 = x0_base
    y0 = y0_base
    xp = xp_base
    yp = yp_base
    
    scores = np.zeros((20, 20), dtype=np.float32)
    for ri in range(20):
        gy = y0 + ri * yp
        for ci in range(20):
            gx = x0 + ci * xp
            pr = 4
            y1 = max(0, int(gy)-pr); y2 = min(h, int(gy)+pr+1)
            x1 = max(0, int(gx)-pr); x2 = min(w, int(gx)+pr+1)
            patch = resp[y1:y2, x1:x2]
            scores[ri, ci] = float(patch.max()) if patch.size else 0.0
    
    print(f"Score stats: min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}")
    print(f"Score per row: {[f'{scores[r,:].mean():.2f}' for r in range(20)]}")
    print(f"Score per col: {[f'{scores[:,c].mean():.2f}' for c in range(20)]}")
    
    # Show scores heatmap
    scores_vis = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scores_big = cv2.resize(scores_vis, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("debug/cropped_grid_scores.png", scores_big)
    print("Saved scores heatmap to debug/cropped_grid_scores.png")
    
    # Try extreme thresholds
    for sq in np.arange(0.05, 0.95, 0.02):
        tv = float(np.quantile(scores, sq))
        bits = (scores >= tv).astype(np.uint8)
        bits[0, :] = 1; bits[:, 0] = 1
        _, oriented = _orient_bits(bits)
        rendered = _render_bits(oriented)
        decoded = _decode_pure_render(rendered)
        if decoded:
            print(f"DECODED sq={sq:.2f}: {decoded}")
            cv2.imwrite("debug/cropped_extreme_DECODED.png", rendered)
            break

print("Done!")
