"""Binary threshold morphsub -> connected components -> RANSAC grid fit -> decode."""
import cv2
import numpy as np
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape

# Step 1: Get clean binary from morphsub
# The dots in morphsub_31 have halos - use a threshold that keeps only the bright parts
for thresh_val in range(30, 120, 5):
    _, binary = cv2.threshold(resp_img, thresh_val, 255, cv2.THRESH_BINARY)
    # Count connected components of reasonable size
    cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    
    # Dot size estimate: pitch ~22px, dot diameter ~12px, area ~113
    valid = [(int(stats[i, cv2.CC_STAT_AREA]), cents[i]) 
             for i in range(1, cc) 
             if 20 <= stats[i, cv2.CC_STAT_AREA] <= 1000]
    
    if 150 <= len(valid) <= 500:
        print(f"thresh={thresh_val}: {len(valid)} components")

# Use best threshold
print("\nSearching for best threshold...")
best_thresh = 60
best_pts = None
best_score = 0

for thresh_val in range(25, 150, 5):
    _, binary = cv2.threshold(resp_img, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphological opening to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary_clean, 8)
    
    # Keep only dot-sized components
    pts = []
    for i in range(1, cc):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh_val = stats[i, cv2.CC_STAT_HEIGHT]
        if 30 <= area <= 800:
            cx, cy = cents[i]
            pts.append((float(cx), float(cy)))
    
    # Score: want ~250-380 points (20x20 grid with ~50-60% fill + borders)
    # and want them to be somewhat regularly spaced
    if len(pts) < 150 or len(pts) > 500:
        continue
    
    arr = np.array(pts)
    xs = arr[:, 0]
    ys = arr[:, 1]
    
    # Simple regularity score: check if nearest-neighbor distances cluster
    from scipy.spatial import KDTree
    tree = KDTree(arr)
    nn = []
    for i in range(min(len(arr), 100)):
        d, _ = tree.query(arr[i], k=min(5, len(arr)))
        nn.extend(d[1:])
    nn = np.array(nn)
    nn = nn[(nn > 10) & (nn < 50)]
    
    if len(nn) < 50:
        continue
    
    # Peak in NN distances = pitch
    hist, bins = np.histogram(nn, bins=40)
    peak_idx = np.argmax(hist)
    pitch_est = (bins[peak_idx] + bins[peak_idx+1]) / 2
    
    # How many NNs are within 20% of pitch?
    on_pitch = np.sum(np.abs(nn - pitch_est) < pitch_est * 0.2)
    score = on_pitch
    
    if score > best_score:
        best_score = score
        best_thresh = thresh_val
        best_pts = pts
        best_pitch = pitch_est

print(f"Best thresh={best_thresh}: {len(best_pts)} pts, pitch={best_pitch:.1f}, score={best_score}")

if best_pts is None:
    print("No good threshold found")
    sys.exit(1)

# Visualize
vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
for x, y in best_pts:
    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), 1)
    cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
cv2.imwrite("debug/cropped_best_pts.png", vis)

arr = np.array(best_pts)
pitch = best_pitch
print(f"\nUsing {len(best_pts)} points, pitch={pitch:.1f}")

# RANSAC-style grid fit
# For each pair of nearby points, compute the grid origin and pitch
# then count inliers

best_result = None
best_inliers = 0

# Try multiple origin candidates
xs = arr[:, 0]
ys = arr[:, 1]

x_range = xs.max() - xs.min()
y_range = ys.max() - ys.min()

print(f"Point bounds: x=[{xs.min():.0f},{xs.max():.0f}] y=[{ys.min():.0f},{ys.max():.0f}]")
print(f"Range: {x_range:.0f} x {y_range:.0f}, expected: {pitch*19:.0f}")

for x0 in np.arange(xs.min() - pitch, xs.min() + pitch/2, pitch * 0.08):
    for y0 in np.arange(ys.min() - pitch, ys.min() + pitch/2, pitch * 0.08):
        for pt in np.arange(pitch * 0.9, pitch * 1.1, pitch * 0.02):
            bits = np.zeros((20, 20), dtype=np.uint8)
            inliers = 0
            
            for x, y in best_pts:
                col = round((x - x0) / pt)
                row = round((y - y0) / pt)
                if 0 <= col < 20 and 0 <= row < 20:
                    ex = x0 + col * pt
                    ey = y0 + row * pt
                    if np.hypot(x - ex, y - ey) < pt * 0.38:
                        bits[row, col] = 1
                        inliers += 1
            
            if inliers < best_inliers:
                continue
            
            best_inliers = inliers
            
            for r in range(4):
                tb = np.rot90(bits, r).copy()
                tb[0, :] = 1; tb[:, 0] = 1
                rendered = _render_bits(tb)
                decoded = _decode_pure_render(rendered)
                if decoded:
                    print(f"DECODED! x0={x0:.0f} y0={y0:.0f} pt={pt:.1f} rot={r*90} inliers={inliers}: {decoded}")
                    cv2.imwrite("debug/cropped_ransac_DECODED.png", rendered)
                    best_result = decoded
                    break
            if best_result: break
        if best_result: break
    if best_result: break

print(f"\nMax inliers: {best_inliers}")
if not best_result:
    print("No decode. Saving best matrix for inspection...")
    # Find x0, y0, pt with best_inliers
    for x0 in np.arange(xs.min() - pitch, xs.min() + pitch/2, pitch * 0.1):
        for y0 in np.arange(ys.min() - pitch, ys.min() + pitch/2, pitch * 0.1):
            bits = np.zeros((20, 20), dtype=np.uint8)
            inliers = 0
            for x, y in best_pts:
                col = round((x - x0) / pitch)
                row = round((y - y0) / pitch)
                if 0 <= col < 20 and 0 <= row < 20:
                    ex = x0 + col * pitch
                    ey = y0 + row * pitch
                    if np.hypot(x - ex, y - ey) < pitch * 0.38:
                        bits[row, col] = 1
                        inliers += 1
            if inliers >= best_inliers - 3:
                # Save visualization with grid
                vis2 = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                for ri in range(20):
                    for ci in range(20):
                        gx = int(x0 + ci * pitch)
                        gy = int(y0 + ri * pitch)
                        color = (0, 0, 255) if bits[ri, ci] else (100, 100, 100)
                        cv2.circle(vis2, (gx, gy), 3, color, -1)
                cv2.imwrite("debug/cropped_grid_best.png", vis2)
                
                for r in range(4):
                    tb = np.rot90(bits, r).copy()
                    tb[0, :] = 1; tb[:, 0] = 1
                    cv2.imwrite(f"debug/cropped_matrix_r{r*90}.png", _render_bits(tb))
                print(f"Saved: x0={x0:.0f} y0={y0:.0f} inliers={inliers}")
                best_inliers = inliers  # update so we only save the best
                break
        else:
            continue
        break

print("Done!")
