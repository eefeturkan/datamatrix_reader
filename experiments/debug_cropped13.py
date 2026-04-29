"""Union multiple dot detectors to maximize coverage, then grid-fit and reconstruct."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render
import zxingcpp

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

# --- Multiple dot detection methods ---
smooth = cv2.bilateralFilter(raw, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)
sharpen = cv2.addWeighted(clahe, 2.0, cv2.GaussianBlur(clahe, (0, 0), 1.5), -1.0, 0)

expected_dot_r = w / 20 * 0.3  # ~7px radius
min_area = 15
max_area = int(3.14 * (expected_dot_r * 2.5) ** 2)
print(f"Dot area range: {min_area}-{max_area}")

all_centroids = set()

def extract_dots(binary, label=""):
    """Extract dot centroids from binary image."""
    cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    dots = []
    for i in range(1, cc):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if min_area <= area <= max_area and aspect > 0.25:
            cx, cy = cents[i]
            dots.append((round(float(cx), 1), round(float(cy), 1)))
    return dots

# Method 1: Tophat
for src_name, src in [("clahe", clahe), ("sharp", sharpen), ("smooth", smooth)]:
    for ks in (9, 11, 13, 15):
        tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dots = extract_dots(binary, f"tophat_{src_name}_k{ks}")
        for d in dots:
            all_centroids.add(d)

# Method 2: Blackhat
for src_name, src in [("clahe", clahe), ("sharp", sharpen)]:
    for ks in (9, 13, 17):
        blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dots = extract_dots(binary)
        for d in dots:
            all_centroids.add(d)

# Method 3: LoG
for sigma in (2.0, 3.0, 4.0):
    blurred = cv2.GaussianBlur(clahe.astype(np.float32), (0, 0), sigma)
    log = -cv2.Laplacian(blurred, cv2.CV_32F)
    log_pos = np.clip(log, 0, None)
    norm = cv2.normalize(log_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    dots = extract_dots(binary)
    for d in dots:
        all_centroids.add(d)

# Method 4: Adaptive threshold
for block in (21, 31, 41):
    for c_val in (3, 5, 8):
        adapt = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, c_val)
        # Only keep small components (dots)
        dots = extract_dots(255 - adapt)
        for d in dots:
            all_centroids.add(d)

print(f"Total unique centroids (raw): {len(all_centroids)}")

# Deduplicate nearby centroids
centroids_list = sorted(all_centroids)
merged = []
used = set()
for i, (x1, y1) in enumerate(centroids_list):
    if i in used:
        continue
    group = [(x1, y1)]
    for j, (x2, y2) in enumerate(centroids_list[i+1:], i+1):
        if j in used:
            continue
        if abs(x2 - x1) < 6 and abs(y2 - y1) < 6:
            group.append((x2, y2))
            used.add(j)
    used.add(i)
    mx = np.mean([p[0] for p in group])
    my = np.mean([p[1] for p in group])
    merged.append((float(mx), float(my)))

print(f"After merging: {len(merged)} centroids")

# Visualize merged centroids
vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
for x, y in merged:
    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), 1)
    cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
cv2.imwrite("debug/cropped_merged_dots.png", vis)

# Create binary from merged centroids
clean = np.zeros_like(raw)
for x, y in merged:
    cv2.circle(clean, (int(x), int(y)), int(expected_dot_r), 255, -1)
cv2.imwrite("debug/cropped_merged_binary.png", clean)

# Try decode on merged binary
print("\n=== Decoding merged binary ===")
for scale in (1, 2, 3, 4):
    scaled = clean if scale == 1 else cv2.resize(clean, None, fx=scale, fy=scale,
                                                   interpolation=cv2.INTER_NEAREST)
    for rot in (0, 90, 180, 270):
        rotated = scaled
        if rot == 90: rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180: rotated = cv2.rotate(scaled, cv2.ROTATE_180)
        elif rot == 270: rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for inv in (False, True):
            test_img = 255 - rotated if inv else rotated
            try:
                results = zxingcpp.read_barcodes(
                    test_img, formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True, try_downscale=True,
                    text_mode=zxingcpp.TextMode.Plain,
                )
                for r in (results or []):
                    text = (getattr(r, "text", "") or "").strip()
                    if text:
                        inv_s = "_inv" if inv else ""
                        print(f"  DECODED! x{scale}_r{rot}{inv_s}: {text}")
                        cv2.imwrite("debug/cropped_merged_decoded.png", test_img)
            except:
                pass

# --- Grid-fit approach using centroids ---
print("\n=== Grid fitting from centroids ===")

# Estimate pitch from nearest-neighbor distances
from scipy.spatial import KDTree
pts = np.array(merged)
tree = KDTree(pts)
nn_dists = []
for i in range(len(pts)):
    d, _ = tree.query(pts[i], k=5)
    nn_dists.extend(d[1:])
nn_dists = np.array(nn_dists)
nn_dists = nn_dists[(nn_dists > 8) & (nn_dists < 40)]

hist, bins = np.histogram(nn_dists, bins=80)
peak_bin = np.argmax(hist)
pitch_est = (bins[peak_bin] + bins[peak_bin+1]) / 2
print(f"Estimated pitch: {pitch_est:.1f}")

# Create a grid-based binary
# Find the bounding box of dots
xs = pts[:, 0]
ys = pts[:, 1]
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
print(f"Dot bounds: x=[{x_min:.0f},{x_max:.0f}] y=[{y_min:.0f},{y_max:.0f}]")
print(f"Expected grid span: {pitch_est * 19:.0f}px for 20 modules")

# Try different grid origins and orientations
module_count = 20
best_decoded = None
best_score = -1

for x_start in np.arange(max(0, x_min - pitch_est), x_min + pitch_est, pitch_est * 0.1):
    for y_start in np.arange(max(0, y_min - pitch_est), y_min + pitch_est, pitch_est * 0.1):
        for pitch_try in np.arange(pitch_est - 3, pitch_est + 3, 0.5):
            # Check how many dots fall on grid
            matched = 0
            bits = np.zeros((module_count, module_count), dtype=np.uint8)
            
            for x, y in merged:
                col = round((x - x_start) / pitch_try)
                row = round((y - y_start) / pitch_try)
                if 0 <= col < module_count and 0 <= row < module_count:
                    dist = np.sqrt((x - (x_start + col * pitch_try))**2 + 
                                   (y - (y_start + row * pitch_try))**2)
                    if dist < pitch_try * 0.4:
                        bits[row, col] = 1
                        matched += 1
            
            if matched < 100:
                continue
            
            # Set solid borders
            bits[0, :] = 1
            bits[:, 0] = 1
            
            orient_score, oriented = _orient_bits(bits)
            occupancy = float(bits.mean())
            score = matched + orient_score * 10
            
            rendered = _render_bits(oriented)
            decoded = _decode_pure_render(rendered)
            if decoded:
                if score > best_score:
                    best_score = score
                    best_decoded = decoded
                    print(f"  DECODED! x0={x_start:.0f} y0={y_start:.0f} "
                          f"pitch={pitch_try:.1f} matched={matched}: {decoded}")
                    cv2.imwrite("debug/cropped_grid_decoded.png", rendered)

if best_decoded:
    print(f"\nBest decode: {best_decoded}")
else:
    print("\nNo decode from grid fitting")
    # Save best bits for visualization
    # Try with forced solid borders on all four possibilities
    for x_start in np.arange(x_min - pitch_est * 0.5, x_min + pitch_est, pitch_est * 0.2):
        for y_start in np.arange(y_min - pitch_est * 0.5, y_min + pitch_est, pitch_est * 0.2):
            bits = np.zeros((module_count, module_count), dtype=np.uint8)
            matched = 0
            for x, y in merged:
                col = round((x - x_start) / pitch_est)
                row = round((y - y_start) / pitch_est)
                if 0 <= col < module_count and 0 <= row < module_count:
                    dist = np.sqrt((x - (x_start + col * pitch_est))**2 + 
                                   (y - (y_start + row * pitch_est))**2)
                    if dist < pitch_est * 0.4:
                        bits[row, col] = 1
                        matched += 1
            
            if matched > 120:
                for r in range(4):
                    test_bits = np.rot90(bits, r).copy()
                    test_bits[0, :] = 1
                    test_bits[:, 0] = 1
                    rendered = _render_bits(test_bits)
                    decoded = _decode_pure_render(rendered)
                    if decoded:
                        print(f"  DECODED (rot{r*90})! x0={x_start:.0f} y0={y_start:.0f} "
                              f"matched={matched}: {decoded}")
                        cv2.imwrite("debug/cropped_grid_rot_decoded.png", rendered)

print("\nDone!")
