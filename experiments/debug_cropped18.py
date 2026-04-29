"""The core issue: _fill_reconstruction_scores uses 7x9 pixel patches.
For cropped.png at full size the dots are ~15px diameter, so the patch 
needs to be larger. Also, the response map doesn't cleanly separate dots.

Solution: Pre-smooth with bilateral, scale down, then use a 
BETTER response map (median background subtraction) that gives 
clean dot centers."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _find_peaks_1d, _fit_progression_subset,
    _orient_bits, _render_bits, _decode_pure_render,
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Original: {w}x{h}")

# Step 1: Heavy bilateral filtering to remove texture while preserving dots
smooth = cv2.bilateralFilter(raw, 15, 100, 100)
smooth = cv2.bilateralFilter(smooth, 15, 100, 100)  # double pass

# Step 2: Downscale to bring pitch into ~12-14px range  
target_w = 280
scale = target_w / w
resized = cv2.resize(smooth, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
rh, rw = resized.shape
print(f"Resized: {rw}x{rh}")

# Step 3: Build a CLEAN response using median background subtraction
# This works because dots are small bright/dark features on smooth background
for median_k in (15, 21, 27, 31):
    bg = cv2.medianBlur(resized, median_k)
    
    # Bright dots (dot brighter than background)
    bright = np.clip(resized.astype(np.float32) - bg.astype(np.float32), 0, None)
    bright_norm = cv2.normalize(bright, None, 0, 1, cv2.NORM_MINMAX)
    
    # Dark dots (dot darker than background = shadow around dot)
    dark = np.clip(bg.astype(np.float32) - resized.astype(np.float32), 0, None)
    dark_norm = cv2.normalize(dark, None, 0, 1, cv2.NORM_MINMAX)
    
    # Combined
    combined = bright_norm + dark_norm
    comb_norm = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
    
    cv2.imwrite(f"debug/cropped_medsub_bright_{median_k}.png", (bright_norm * 255).astype(np.uint8))
    cv2.imwrite(f"debug/cropped_medsub_dark_{median_k}.png", (dark_norm * 255).astype(np.uint8))
    cv2.imwrite(f"debug/cropped_medsub_comb_{median_k}.png", (comb_norm * 255).astype(np.uint8))
    
    for resp_name, resp in [("bright", bright_norm), ("dark", dark_norm), ("comb", comb_norm)]:
        strip_height = max(16, min(28, rh // 12))
        min_distance = max(6, rw // 36)
        
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for pq in np.arange(0.40, 0.85, 0.05):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            
            if not (18 <= len(peaks) <= 22):
                continue
            
            diffs = np.diff(peaks)
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
            if len(valid_diffs) < 6:
                continue
            
            base_pitch = float(np.median(valid_diffs))
            if not 10.0 <= base_pitch <= 16.0:
                continue
            
            fitted = _fit_progression_subset(
                np.array(peaks, dtype=np.float32).tolist(), 20, base_pitch
            )
            if fitted is None:
                continue
            
            print(f"\n  FIT OK: medk={median_k} {resp_name} pq={pq:.2f} peaks={len(peaks)} pitch={base_pitch:.1f}")
            print(f"    fitted diffs: {[f'{d:.0f}' for d in np.diff(fitted)]}")
            
            top_centers = np.array(fitted, dtype=np.float32)
            
            # Try reconstruction with VARIABLE patch sizes
            for patch_r in (3, 4, 5, 6):
                for vo in range(2, 40, 2):
                    for shear in np.arange(-2.0, 2.1, 0.4):
                        for pd in np.arange(-1.0, 1.1, 0.3):
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
                                    y0 = max(0, cy - patch_r)
                                    y1 = min(resp.shape[0], cy + patch_r + 1)
                                    x0 = max(0, cx - patch_r)
                                    x1 = min(resp.shape[1], cx + patch_r + 1)
                                    patch = resp[y0:y1, x0:x1]
                                    scores[row, col] = float(patch.max()) if patch.size else 0.0
                            
                            if not valid:
                                continue
                            
                            for sq in (0.35, 0.40, 0.45, 0.50, 0.55):
                                tv = float(np.quantile(scores, sq))
                                bits = (scores >= tv).astype(np.uint8)
                                bits[0, :] = 1
                                bits[:, 0] = 1
                                orient_score, oriented = _orient_bits(bits)
                                
                                rendered = _render_bits(oriented)
                                decoded = _decode_pure_render(rendered)
                                if decoded:
                                    print(f"    DECODED! patch_r={patch_r} vo={vo} shear={shear:.1f} "
                                          f"pd={pd:.1f} sq={sq:.2f}: {decoded}")
                                    cv2.imwrite(f"debug/cropped_medsub_decoded.png", rendered)
                                    cv2.imwrite(f"debug/cropped_medsub_resp.png", (resp * 255).astype(np.uint8))

# Also try standard pipeline responses on the smoothed+downscaled version
print("\n\n=== Standard responses on smoothed+downscaled ===")
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(resized)
sharpen_r = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)

for elem_k in (7, 9, 11):
    tophat = cv2.morphologyEx(
        sharpen_r, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elem_k, elem_k))
    ).astype(np.float32)
    resp = cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)
    
    strip_height = max(16, min(28, rh // 12))
    min_distance = max(6, rw // 36)
    top_proj = resp[:strip_height, :].mean(axis=0)
    
    for pq in np.arange(0.45, 0.80, 0.05):
        threshold = float(np.quantile(top_proj, pq))
        peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
        
        if not (18 <= len(peaks) <= 22):
            continue
        
        diffs = np.diff(peaks)
        valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
        if len(valid_diffs) < 6:
            continue
        
        base_pitch = float(np.median(valid_diffs))
        if not 10.0 <= base_pitch <= 16.0:
            continue
        
        fitted = _fit_progression_subset(
            np.array(peaks, dtype=np.float32).tolist(), 20, base_pitch
        )
        if fitted is None:
            continue
        
        print(f"\n  FIT: tophat_k{elem_k} pq={pq:.2f} peaks={len(peaks)} pitch={base_pitch:.1f}")
        top_centers = np.array(fitted, dtype=np.float32)
        
        for patch_r in (3, 4, 5):
            for vo in range(2, 35, 2):
                for shear in np.arange(-2.0, 2.1, 0.5):
                    for pd in np.arange(-0.5, 0.6, 0.2):
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
                                y0, y1 = max(0, cy-patch_r), min(resp.shape[0], cy+patch_r+1)
                                x0, x1 = max(0, cx-patch_r), min(resp.shape[1], cx+patch_r+1)
                                patch = resp[y0:y1, x0:x1]
                                scores[row, col] = float(patch.max()) if patch.size else 0.0
                        if not valid: continue
                        
                        for sq in (0.40, 0.45, 0.50, 0.55):
                            tv = float(np.quantile(scores, sq))
                            bits = (scores >= tv).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            _, oriented = _orient_bits(bits)
                            rendered = _render_bits(oriented)
                            decoded = _decode_pure_render(rendered)
                            if decoded:
                                print(f"    DECODED! pr={patch_r} vo={vo} s={shear:.1f} pd={pd:.1f} sq={sq:.2f}: {decoded}")
                                cv2.imwrite(f"debug/cropped_std_decoded.png", rendered)

print("\nDone!")
