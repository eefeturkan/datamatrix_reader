"""Direct approach: HoughCircles on morphsub_31 -> grid fit -> decode."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape
print(f"Image: {w}x{h}")

# Use HoughCircles on morphsub - the dots are clearly circular there
# Tune for the actual dot size in this image
circles_best = None
best_n = 0
for dp in (1.0, 1.5):
    for minD in (15, 18, 20, 22):
        for p1 in (50, 80, 100):
            for p2 in (15, 20, 25, 30):
                for (mr, MR) in [(5, 12), (6, 13), (7, 14), (8, 15)]:
                    circles = cv2.HoughCircles(
                        resp_img, cv2.HOUGH_GRADIENT,
                        dp=dp, minDist=minD,
                        param1=p1, param2=p2,
                        minRadius=mr, maxRadius=MR
                    )
                    if circles is not None:
                        n = len(circles[0])
                        # Want 250-430 circles (variable fill + borders)
                        # but let's be generous: 180-450
                        if 200 <= n <= 450 and abs(n - 330) < abs(best_n - 330):
                            best_n = n
                            circles_best = circles[0].copy()

if circles_best is None:
    print("No Hough circles found in right range, trying wider...")
    for dp in (1.0, 2.0):
        for minD in (12, 15, 18):
            for p2 in (8, 10, 12):
                circles = cv2.HoughCircles(
                    resp_img, cv2.HOUGH_GRADIENT,
                    dp=dp, minDist=minD,
                    param1=50, param2=p2,
                    minRadius=4, maxRadius=16
                )
                if circles is not None:
                    n = len(circles[0])
                    if 150 <= n <= 500 and abs(n - 330) < abs(best_n - 330):
                        best_n = n
                        circles_best = circles[0].copy()

print(f"Best Hough: {len(circles_best) if circles_best is not None else 0} circles")

if circles_best is not None:
    # Visualize
    vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for c in circles_best:
        cv2.circle(vis, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 1)
        cv2.circle(vis, (int(c[0]), int(c[1])), 2, (0, 0, 255), -1)
    cv2.imwrite("debug/cropped_hough_morphsub.png", vis)
    
    pts = [(float(c[0]), float(c[1])) for c in circles_best]
    
    # Estimate pitch from nearest-neighbor
    from scipy.spatial import KDTree
    arr = np.array(pts)
    tree = KDTree(arr)
    nn = []
    for i in range(len(arr)):
        d, _ = tree.query(arr[i], k=min(5, len(arr)))
        nn.extend(d[1:])
    nn = np.array(nn)
    nn = nn[(nn > 12) & (nn < 50)]
    hist, bins = np.histogram(nn, bins=60)
    pitch = float((bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2)
    print(f"Estimated pitch: {pitch:.1f}")
    
    # Grid fitting: brute force on origin + pitch
    xs = arr[:, 0]
    ys = arr[:, 1]
    
    best_result = None
    best_matched = 0
    
    for x0 in np.arange(xs.min() - pitch, xs.min() + pitch, pitch * 0.1):
        for y0 in np.arange(ys.min() - pitch, ys.min() + pitch, pitch * 0.1):
            for pt in np.arange(pitch * 0.85, pitch * 1.15, pitch * 0.02):
                bits = np.zeros((20, 20), dtype=np.uint8)
                matched = 0
                for x, y in pts:
                    col = round((x - x0) / pt)
                    row = round((y - y0) / pt)
                    if 0 <= col < 20 and 0 <= row < 20:
                        ex = x0 + col * pt
                        ey = y0 + row * pt
                        if np.hypot(x - ex, y - ey) < pt * 0.4:
                            bits[row, col] = 1
                            matched += 1
                
                if matched <= best_matched:
                    continue
                
                best_matched = matched
                # Try all 4 rotations
                for r in range(4):
                    tb = np.rot90(bits, r).copy()
                    tb[0, :] = 1; tb[:, 0] = 1
                    rendered = _render_bits(tb)
                    decoded = _decode_pure_render(rendered)
                    if decoded:
                        print(f"DECODED! x0={x0:.0f} y0={y0:.0f} pt={pt:.1f} rot={r*90} matched={matched}: {decoded}")
                        cv2.imwrite("debug/cropped_hough_DECODED.png", rendered)
                        best_result = decoded
                        break
                if best_result:
                    break
            if best_result:
                break
        if best_result:
            break
    
    print(f"Best matched: {best_matched}")
    if not best_result:
        print("No decode. Let's inspect the best bits matrix...")
        # Try forced decode with best_matched grid
        for x0 in np.arange(xs.min() - pitch, xs.min() + pitch, pitch * 0.15):
            for y0 in np.arange(ys.min() - pitch, ys.min() + pitch, pitch * 0.15):
                bits = np.zeros((20, 20), dtype=np.uint8)
                matched = 0
                for x, y in pts:
                    col = round((x - x0) / pitch)
                    row = round((y - y0) / pitch)
                    if 0 <= col < 20 and 0 <= row < 20:
                        ex = x0 + col * pitch
                        ey = y0 + row * pitch
                        if np.hypot(x - ex, y - ey) < pitch * 0.4:
                            bits[row, col] = 1
                            matched += 1
                
                if matched > best_matched - 5:
                    # Save one rendering for visual inspection
                    for r in range(4):
                        tb = np.rot90(bits, r).copy()
                        tb[0, :] = 1; tb[:, 0] = 1
                        rendered = _render_bits(tb)
                        cv2.imwrite(f"debug/cropped_bits_r{r*90}.png", rendered)
                    best_matched = matched
                    print(f"Saved bits with {matched} matched at x0={x0:.0f} y0={y0:.0f}")
                    break
            else:
                continue
            break

print("Done!")
