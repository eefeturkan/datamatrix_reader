"""Focused approach: isolate circular dots via HoughCircles + create synthetic binary."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render
import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

smoothed = cv2.bilateralFilter(raw, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smoothed)

# ============================================================
# Approach 1: Hough Circle Detection
# ============================================================
print("\n=== HoughCircles approach ===")
best_circles = None
best_count = 0

for dp in (1.0, 1.5, 2.0):
    for min_dist in (12, 15, 18, 20):
        for param1 in (30, 50, 70):
            for param2 in (8, 12, 16, 20):
                for min_r, max_r in [(3, 8), (4, 10), (5, 12), (6, 14)]:
                    for src_name, src in [("clahe", clahe), ("smooth", smoothed)]:
                        circles = cv2.HoughCircles(
                            src, cv2.HOUGH_GRADIENT, dp=dp,
                            minDist=min_dist,
                            param1=param1, param2=param2,
                            minRadius=min_r, maxRadius=max_r
                        )
                        if circles is not None:
                            count = len(circles[0])
                            # We expect ~250-400 dots for 20x20 data matrix (depending on data)
                            # At minimum ~200 (solid borders = 20+19+19+18 = 76 dots)
                            if 150 <= count <= 450 and count > best_count:
                                best_count = count
                                best_circles = circles[0]
                                print(f"  dp={dp} minD={min_dist} p1={param1} p2={param2} "
                                      f"r={min_r}-{max_r} {src_name}: {count} circles")

if best_circles is not None:
    print(f"\nBest: {len(best_circles)} circles")
    
    # Visualize
    vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for c in best_circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 1, (0, 0, 255), -1)
    cv2.imwrite("debug/cropped_hough_circles.png", vis)
    
    # Create binary image from detected circles
    binary = np.zeros_like(raw)
    for c in best_circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        cv2.circle(binary, (cx, cy), max(r, 3), 255, -1)
    
    cv2.imwrite("debug/cropped_hough_binary.png", binary)
    
    # Try decode on this binary
    for scale in (2, 3, 4):
        scaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for rot in (0, 90, 180, 270):
            rotated = scaled
            if rot == 90: rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: rotated = cv2.rotate(scaled, cv2.ROTATE_180)
            elif rot == 270: rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Try both polarities
            for inv_name, img in [("normal", rotated), ("inv", 255-rotated)]:
                try:
                    results = zxingcpp.read_barcodes(
                        img, formats=zxingcpp.BarcodeFormat.DataMatrix,
                        try_rotate=True, try_downscale=True,
                        text_mode=zxingcpp.TextMode.Plain,
                    )
                    for r in (results or []):
                        text = (getattr(r, "text", "") or "").strip()
                        if text:
                            print(f"  DECODED! x{scale} r{rot} {inv_name}: {text}")
                            cv2.imwrite(f"debug/cropped_hough_decoded.png", img)
                except:
                    pass

# ============================================================
# Approach 2: Grid-based dot detection using refined response
# The key insight: the reconstruction pipeline's _fill_reconstruction_scores
# uses a 7x9 patch (hardcoded small), but for this image the dots are larger.
# Let's adapt the patch size to the actual dot size.
# ============================================================
print("\n=== Approach 2: Adapted patch-size reconstruction ===")

# Build a better response with bilateral + tophat
for elem_size in (7, 9, 11, 13, 15):
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elem_size, elem_size))
    ).astype(np.float32)
    blackhat = cv2.morphologyEx(
        sharpen, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elem_size, elem_size))
    ).astype(np.float32)
    
    for resp_name, resp_raw in [("tophat", tophat), ("blackhat", blackhat),
                                 ("mix", 0.65*tophat + 0.35*blackhat)]:
        resp = cv2.normalize(resp_raw, None, 0, 1, cv2.NORM_MINMAX)
        
        # Find peaks in top strip with various settings
        strip_height = max(16, min(40, h // 10))
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for min_dist in (8, 10, 12, 15, 18):
            for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
                threshold = float(np.quantile(top_proj, pq))
                peaks = []
                for idx in range(1, len(top_proj) - 1):
                    cur = float(top_proj[idx])
                    if cur < threshold or cur < top_proj[idx-1] or cur <= top_proj[idx+1]:
                        continue
                    if peaks and idx - peaks[-1] < min_dist:
                        if cur > float(top_proj[peaks[-1]]):
                            peaks[-1] = idx
                    else:
                        peaks.append(idx)
                
                if not (18 <= len(peaks) <= 22):
                    continue
                
                diffs = np.diff(peaks)
                valid_diffs = diffs[(diffs > 6.0) & (diffs < 35.0)]
                if len(valid_diffs) < 8:
                    continue
                
                base_pitch = float(np.median(valid_diffs))
                if base_pitch < 8 or base_pitch > 30:
                    continue
                
                # Extended fit
                values = sorted(float(p) for p in peaks)
                best_score = -1
                best_progression = None
                for start in values:
                    for pd in (-2.0, -1.0, 0.0, 1.0, 2.0):
                        pitch_try = base_pitch + pd
                        if pitch_try < 6:
                            continue
                        expected = [start + i * pitch_try for i in range(20)]
                        matched = sum(1 for e in expected 
                                      if min(abs(v - e) for v in values) <= pitch_try * 0.4)
                        if matched > best_score:
                            best_score = matched
                            best_progression = []
                            for e in expected:
                                nearest = min(values, key=lambda v: abs(v - e))
                                if abs(nearest - e) <= pitch_try * 0.4:
                                    best_progression.append(nearest)
                                else:
                                    best_progression.append(e)
                
                if best_score < 15:
                    continue
                
                top_centers = np.array(best_progression, dtype=np.float32)
                
                # Try reconstruction with ADAPTED patch size
                patch_radius = max(3, int(base_pitch * 0.25))
                
                for vo in range(2, int(base_pitch * 2.5) + 1, 2):
                    for shear in np.arange(-2.0, 2.1, 0.5):
                        for pd in (-0.5, 0.0, 0.5):
                            pitch = base_pitch + pd
                            scores = np.zeros((20, 20), dtype=np.float32)
                            valid = True
                            
                            for row in range(20):
                                cy = int(round(vo + row * pitch))
                                if cy < 0 or cy >= resp.shape[0]:
                                    valid = False
                                    break
                                row_centers = top_centers + row * shear
                                if row_centers.min() < 0 or row_centers.max() >= resp.shape[1]:
                                    valid = False
                                    break
                                for col, cx_f in enumerate(row_centers):
                                    cx = int(round(float(cx_f)))
                                    y0 = max(0, cy - patch_radius)
                                    y1 = min(resp.shape[0], cy + patch_radius + 1)
                                    x0 = max(0, cx - patch_radius)
                                    x1 = min(resp.shape[1], cx + patch_radius + 1)
                                    patch = resp[y0:y1, x0:x1]
                                    scores[row, col] = float(patch.max()) if patch.size else 0.0
                            
                            if not valid:
                                continue
                            
                            for sq in (0.40, 0.45, 0.50, 0.55):
                                tv = float(np.quantile(scores, sq))
                                bits = (scores >= tv).astype(np.uint8)
                                bits[0, :] = 1
                                bits[:, 0] = 1
                                orient_score, oriented = _orient_bits(bits)
                                occupancy = float(bits.mean())
                                score = orient_score - abs(occupancy - 0.5) * 1.2
                                
                                rendered = _render_bits(oriented)
                                decoded = _decode_pure_render(rendered)
                                if decoded:
                                    print(f"  DECODED! elem={elem_size} {resp_name} md={min_dist} "
                                          f"pq={pq:.2f} pitch={base_pitch:.1f} vo={vo} "
                                          f"shear={shear:.1f} sq={sq:.2f}: {decoded}")
                                    cv2.imwrite("debug/cropped_adapted_decoded.png", rendered)
                                    # Don't break - collect all candidates
                
                # Only need first working configuration per response
                break

print("\nDone!")
