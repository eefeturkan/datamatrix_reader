"""175 noktayla kesin grid fit. Solid border bilgisini kullan."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape

# Get the 175 points from thresh=95 + morphOpen
_, binary = cv2.threshold(resp_img, 95, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary_clean, 8)
pts = []
for i in range(1, cc):
    area = stats[i, cv2.CC_STAT_AREA]
    if 30 <= area <= 800:
        pts.append((float(cents[i][0]), float(cents[i][1])))
print(f"Points: {len(pts)}")

arr = np.array(pts)
xs = arr[:, 0]
ys = arr[:, 1]

# From visual inspection: top-left corner of grid is around (63, 18)
# Grid spans x:[63,388] y:[18,366] -> span ~325x348
# 19 intervals -> pitch = 325/19 = 17.1 x, 348/19 = 18.3 y
# Let's be more precise - the grid is slightly distorted (shear/rotation)

# Step 1: Find the top-row points (smallest y values)
# Sort by y, take top cluster
sorted_by_y = sorted(pts, key=lambda p: p[1])
top_y = sorted_by_y[0][1]
# Get all points within 10px of top row
top_row = [(x, y) for x, y in pts if y < top_y + 15]
top_row.sort(key=lambda p: p[0])
print(f"Top row: {len(top_row)} points")
print(f"Top row x: {[f'{x:.0f}' for x, y in top_row]}")

# Step 2: Find left-column points
sorted_by_x = sorted(pts, key=lambda p: p[0])
left_x = sorted_by_x[0][0]
left_col = [(x, y) for x, y in pts if x < left_x + 15]
left_col.sort(key=lambda p: p[1])
print(f"Left col: {len(left_col)} points")
print(f"Left col y: {[f'{y:.0f}' for x, y in left_col]}")

# Estimate pitch from top row and left col
if len(top_row) >= 3:
    top_x_diffs = np.diff([x for x, y in top_row])
    x_pitch = float(np.median(top_x_diffs))
    print(f"X pitch from top row: {x_pitch:.1f}")
else:
    x_pitch = 17.5

if len(left_col) >= 3:
    left_y_diffs = np.diff([y for x, y in left_col])
    y_pitch = float(np.median(left_y_diffs))
    print(f"Y pitch from left col: {y_pitch:.1f}")
else:
    y_pitch = 17.5

pitch = (x_pitch + y_pitch) / 2
print(f"Average pitch: {pitch:.1f}")

# Grid origin
x0 = top_row[0][0] if top_row else xs.min()
y0 = left_col[0][1] if left_col else ys.min()
print(f"Grid origin: ({x0:.0f}, {y0:.0f})")

# Step 3: Exhaustive search around this estimate
best_decoded = None
best_inliers = 0
best_bits = None

for dx0 in np.arange(-x_pitch/2, x_pitch/2, 1.0):
    for dy0 in np.arange(-y_pitch/2, y_pitch/2, 1.0):
        for dxp in np.arange(-2.0, 2.1, 0.3):
            for dyp in np.arange(-2.0, 2.1, 0.3):
                ox = x0 + dx0
                oy = y0 + dy0
                xp = x_pitch + dxp
                yp = y_pitch + dyp
                
                bits = np.zeros((20, 20), dtype=np.uint8)
                inliers = 0
                
                for x, y in pts:
                    col = round((x - ox) / xp)
                    row = round((y - oy) / yp)
                    if 0 <= col < 20 and 0 <= row < 20:
                        ex = ox + col * xp
                        ey = oy + row * yp
                        tol = min(xp, yp) * 0.4
                        if abs(x - ex) < tol and abs(y - ey) < tol:
                            bits[row, col] = 1
                            inliers += 1
                
                if inliers < best_inliers:
                    continue
                
                for r in range(4):
                    tb = np.rot90(bits, r).copy()
                    tb[0, :] = 1; tb[:, 0] = 1
                    rendered = _render_bits(tb)
                    decoded = _decode_pure_render(rendered)
                    if decoded:
                        if inliers > best_inliers or best_decoded is None:
                            best_decoded = decoded
                            best_inliers = inliers
                            print(f"DECODED! r={r*90} inliers={inliers} dx={dx0:.0f} dy={dy0:.0f} dxp={dxp:.1f} dyp={dyp:.1f}: {decoded}")
                            cv2.imwrite("debug/cropped_FINAL.png", rendered)
                            
                            # Grid overlay
                            vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                            for ri in range(20):
                                for ci in range(20):
                                    gx = int(ox + ci * xp)
                                    gy = int(oy + ri * yp)
                                    color = (0, 0, 255) if bits[ri, ci] else (100, 100, 100)
                                    cv2.circle(vis, (gx, gy), 3, color, -1)
                            for px, py in pts:
                                cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 0), 1)
                            cv2.imwrite("debug/cropped_FINAL_grid.png", vis)
                        break
                
                if inliers > best_inliers and not best_decoded:
                    best_inliers = inliers
                    best_bits = bits.copy()

print(f"\nMax inliers: {best_inliers}")
if not best_decoded and best_bits is not None:
    print(f"Fill rate: {best_bits.mean():.2%}")
    for r in range(4):
        tb = np.rot90(best_bits, r).copy()
        tb[0, :] = 1; tb[:, 0] = 1
        cv2.imwrite(f"debug/cropped_bestmatrix_r{r*90}.png", _render_bits(tb))
    print("Saved best matrix renders")

print("Done!")
