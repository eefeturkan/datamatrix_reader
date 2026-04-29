"""Use morphsub_31 directly as response map and run reconstruction."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _find_peaks_1d, _fit_progression_subset,
    _fill_reconstruction_scores, _orient_bits, _render_bits, _decode_pure_render,
)

# Load the morphsub_31 image directly as response map
resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
resp = resp_img.astype(np.float32) / 255.0
h, w = resp.shape
print(f"Response map: {w}x{h}")

strip_height = max(16, min(40, h // 10))
min_distance = max(6, w // 40)
print(f"strip_height={strip_height}, min_distance={min_distance}")

top_proj = resp[:strip_height, :].mean(axis=0)

# Find peaks
best_fit = None
for pq in np.arange(0.40, 0.90, 0.02):
    threshold = float(np.quantile(top_proj, pq))
    peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
    
    if not (17 <= len(peaks) <= 23):
        continue
    
    diffs = np.diff(peaks)
    # Extended range - dots could be 8-28px apart
    valid_diffs = diffs[(diffs > 8.0) & (diffs < 28.0)]
    if len(valid_diffs) < 6:
        continue
    
    base_pitch = float(np.median(valid_diffs))
    
    # Try fit with extended tolerance
    values = sorted(float(p) for p in peaks)
    best_score = -1
    best_prog = None
    for start in values:
        for pd in np.arange(-2.0, 2.1, 0.5):
            pitch_try = base_pitch + pd
            if pitch_try < 5:
                continue
            expected = [start + i * pitch_try for i in range(20)]
            matched = 0
            prog = []
            for e in expected:
                nearest = min(values, key=lambda v: abs(v - e))
                if abs(nearest - e) <= pitch_try * 0.4:
                    matched += 1
                    prog.append(nearest)
                else:
                    prog.append(e)
            if matched > best_score:
                best_score = matched
                best_prog = prog
    
    if best_score >= 15:
        if best_fit is None or best_score > best_fit[0]:
            best_fit = (best_score, base_pitch, best_prog, pq, peaks)
        print(f"pq={pq:.2f} peaks={len(peaks)} pitch={base_pitch:.1f} matched={best_score}/20")

if best_fit is None:
    print("No fit found - showing raw peaks at various pq:")
    for pq in np.arange(0.40, 0.90, 0.05):
        t = float(np.quantile(top_proj, pq))
        pks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=t)
        print(f"  pq={pq:.2f}: {len(pks)} peaks")
    sys.exit(1)

best_score, base_pitch, fitted, best_pq, peaks = best_fit
print(f"\nBest fit: pq={best_pq:.2f} pitch={base_pitch:.1f} matched={best_score}/20")
top_centers = np.array(fitted, dtype=np.float32)

# Run reconstruction with this response map
print("\nRunning reconstruction...")
decoded_any = False
for vo in range(1, int(base_pitch * 3) + 1, 1):
    for shear in np.arange(-3.0, 3.1, 0.3):
        for pd in np.arange(-2.0, 2.1, 0.3):
            pitch = base_pitch + pd
            scores = np.zeros((20, 20), dtype=np.float32)
            valid = True
            for row in range(20):
                cy = int(round(vo + row * pitch))
                if cy < 0 or cy >= resp.shape[0]:
                    valid = False; break
                row_centers = top_centers + row * shear
                if row_centers.min() < 0 or row_centers.max() >= resp.shape[1]:
                    valid = False; break
                for col, cx_f in enumerate(row_centers):
                    cx = int(round(float(cx_f)))
                    pr = max(3, int(pitch * 0.25))
                    y0 = max(0, cy - pr); y1 = min(resp.shape[0], cy + pr + 1)
                    x0 = max(0, cx - pr); x1 = min(resp.shape[1], cx + pr + 1)
                    patch = resp[y0:y1, x0:x1]
                    scores[row, col] = float(patch.max()) if patch.size else 0.0
            if not valid:
                continue
            for sq in np.arange(0.35, 0.66, 0.05):
                tv = float(np.quantile(scores, sq))
                bits = (scores >= tv).astype(np.uint8)
                bits[0, :] = 1; bits[:, 0] = 1
                orient_score, oriented = _orient_bits(bits)
                rendered = _render_bits(oriented)
                decoded = _decode_pure_render(rendered)
                if decoded:
                    print(f"DECODED! vo={vo} shear={shear:.1f} pd={pd:.1f} sq={sq:.2f}: {decoded}")
                    cv2.imwrite("debug/cropped_final_SUCCESS.png", rendered)
                    decoded_any = True
                    break
            if decoded_any: break
        if decoded_any: break
    if decoded_any: break

if not decoded_any:
    print("Still no decode. Saving a sample score matrix for inspection.")
    # Save the response and top projection for manual inspection
    cv2.imwrite("debug/cropped_top_strip.png", (resp[:strip_height, :] * 255).astype(np.uint8))
    print(f"Top projection saved. Peaks at pq={best_pq:.2f}: {peaks}")
    print(f"Fitted: {[f'{x:.0f}' for x in fitted]}")

print("Done!")
