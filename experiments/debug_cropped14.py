"""Tighter dot filtering: only keep larger, rounder dots. Use circularity + area."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render
import zxingcpp

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape

smooth = cv2.bilateralFilter(raw, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)

# Real dots are ~6-10px radius, area ~113-314
# Background texture dots are ~3-5px radius, area ~28-78
# So use min_area=60 to exclude most background

def extract_quality_dots(binary, min_area=60, max_area=800, min_circularity=0.4):
    """Extract dots with strict circularity and size filtering."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * 3.14159 * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dots.append((float(cx), float(cy), area, circularity))
    return dots

# Try different preprocessing + strict filtering
all_results = {}

for ks in (9, 11, 13, 15, 17, 19, 21):
    for src_name, src in [("clahe", clahe), ("smooth", smooth)]:
        # Tophat
        tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        for min_a in (40, 60, 80, 100):
            for min_c in (0.3, 0.4, 0.5):
                dots = extract_quality_dots(binary, min_area=min_a, min_circularity=min_c)
                key = f"th{ks}_{src_name}_a{min_a}_c{min_c}"
                if 150 <= len(dots) <= 450:
                    all_results[key] = dots
                    if len(dots) >= 200:
                        print(f"{key}: {len(dots)} dots")

# Also try blackhat
for ks in (9, 13, 17, 21):
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for min_a in (40, 60, 80):
        for min_c in (0.3, 0.4, 0.5):
            dots = extract_quality_dots(binary, min_area=min_a, min_circularity=min_c)
            key = f"bh{ks}_a{min_a}_c{min_c}"
            if 150 <= len(dots) <= 450:
                all_results[key] = dots
                if len(dots) >= 200:
                    print(f"{key}: {len(dots)} dots")

print(f"\nTotal candidates: {len(all_results)}")

# For each good candidate set, try grid fitting and decode
from scipy.spatial import KDTree

best_decode = None
for key, dots in sorted(all_results.items(), key=lambda x: abs(len(x[1]) - 300)):
    pts = np.array([(d[0], d[1]) for d in dots])
    
    # Estimate pitch
    tree = KDTree(pts)
    nn_dists = []
    for i in range(min(len(pts), 200)):
        d, _ = tree.query(pts[i], k=min(5, len(pts)))
        nn_dists.extend(d[1:])
    nn_dists = np.array(nn_dists)
    nn_dists = nn_dists[(nn_dists > 12) & (nn_dists < 50)]
    
    if len(nn_dists) < 20:
        continue
    
    hist, bins = np.histogram(nn_dists, bins=50)
    peak_bin = np.argmax(hist)
    pitch = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    if pitch < 15 or pitch > 35:
        continue
    
    # Grid fit
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    for x0 in np.arange(max(0, xs.min() - pitch), xs.min() + pitch, pitch * 0.15):
        for y0 in np.arange(max(0, ys.min() - pitch), ys.min() + pitch, pitch * 0.15):
            for pt in np.arange(pitch - 2, pitch + 2.1, 0.5):
                bits = np.zeros((20, 20), dtype=np.uint8)
                matched = 0
                for x, y in pts:
                    col = round((x - x0) / pt)
                    row = round((y - y0) / pt)
                    if 0 <= col < 20 and 0 <= row < 20:
                        dist = np.sqrt((x - (x0 + col * pt))**2 + (y - (y0 + row * pt))**2)
                        if dist < pt * 0.4:
                            bits[row, col] = 1
                            matched += 1
                
                if matched < len(dots) * 0.5:
                    continue
                
                # Try all 4 rotations with forced borders
                for r in range(4):
                    test_bits = np.rot90(bits, r).copy()
                    test_bits[0, :] = 1
                    test_bits[:, 0] = 1
                    rendered = _render_bits(test_bits)
                    decoded = _decode_pure_render(rendered)
                    if decoded:
                        print(f"  DECODED! {key} pitch={pt:.1f} x0={x0:.0f} y0={y0:.0f} "
                              f"rot={r*90} matched={matched}: {decoded}")
                        cv2.imwrite(f"debug/cropped_final_decoded.png", rendered)
                        
                        # Visualize
                        vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                        for x, y, _, _ in dots:
                            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), 1)
                        for row_i in range(20):
                            for col_i in range(20):
                                gx = int(x0 + col_i * pt)
                                gy = int(y0 + row_i * pt)
                                color = (0, 0, 255) if bits[row_i, col_i] else (128, 128, 128)
                                cv2.circle(vis, (gx, gy), 2, color, -1)
                        cv2.imwrite("debug/cropped_final_grid.png", vis)
                        
                        best_decode = decoded
                        break
                if best_decode:
                    break
            if best_decode:
                break
        if best_decode:
            break
    if best_decode:
        break

if not best_decode:
    print("No decode achieved.")
    # Show best candidates for manual inspection
    for key, dots in sorted(all_results.items(), key=lambda x: -len(x[1]))[:3]:
        print(f"\n{key}: {len(dots)} dots")
        vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        for x, y, area, circ in dots:
            cv2.circle(vis, (int(x), int(y)), int(np.sqrt(area/3.14)), (0, 255, 0), 1)
        cv2.imwrite(f"debug/cropped_best_{key}.png", vis)

print("\nDone!")
