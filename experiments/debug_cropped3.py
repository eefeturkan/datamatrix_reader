"""Test a better approach for cropped.png - adaptive local dot detection"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _fit_progression_subset, _find_peaks_1d, _orient_bits, 
    _render_bits, _decode_pure_render, _fill_reconstruction_scores,
    _select_top_rendered_candidates, _RenderedCandidate,
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

# Approach: The dots in cropped.png are specular highlights on metal surface.
# They appear as small bright spots. We need a different strategy.
# 
# Strategy: Use morphological closing to estimate the background,
# then subtract to isolate dots, regardless of background brightness.

def build_dot_enhanced_responses(roi):
    """Build responses optimized for bright-background dot-peen images."""
    responses = []
    
    # 1. Standard tophat (for dark-background compatibility)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(roi)
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    
    # 2. Background-adaptive approach: remove low-frequency background
    # Use a large Gaussian blur as background estimate
    for blur_k in (31, 51):
        bg = cv2.GaussianBlur(roi.astype(np.float32), (blur_k, blur_k), 0)
        local = roi.astype(np.float32) - bg
        # Dots could be either brighter or darker than background
        for name, signal in [
            (f"local_bright_{blur_k}", np.clip(local, 0, None)),
            (f"local_dark_{blur_k}", np.clip(-local, 0, None)),
        ]:
            norm = cv2.normalize(signal, None, 0, 1, cv2.NORM_MINMAX)
            if norm is not None:
                responses.append((name, norm))
    
    # 3. LoG (Laplacian of Gaussian) - excellent for blob/dot detection
    for sigma in (2.0, 3.0, 4.0):
        blurred = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), sigma)
        log = -cv2.Laplacian(blurred, cv2.CV_32F)  # negative: bright dots give positive response
        log_pos = np.clip(log, 0, None)
        norm = cv2.normalize(log_pos, None, 0, 1, cv2.NORM_MINMAX)
        if norm is not None:
            responses.append((f"log_{sigma}", norm))
    
    # 4. Also try with CLAHE-enhanced versions
    for sigma in (2.0, 3.0):
        blurred = cv2.GaussianBlur(clahe.astype(np.float32), (0, 0), sigma)
        log = -cv2.Laplacian(blurred, cv2.CV_32F)
        log_pos = np.clip(log, 0, None)
        norm = cv2.normalize(log_pos, None, 0, 1, cv2.NORM_MINMAX)
        if norm is not None:
            responses.append((f"clahe_log_{sigma}", norm))
    
    return responses

responses = build_dot_enhanced_responses(raw)

# Now check peak detection with each response
strip_height = max(16, min(28, h // 12))
min_distance = max(8, w // 36)
module_count = 20

print(f"\nstrip_height: {strip_height}, min_distance: {min_distance}")

for resp_name, resp in responses:
    cv2.imwrite(f"debug/cropped_{resp_name}.png", (resp * 255).astype(np.uint8))
    top_proj = resp[:strip_height, :].mean(axis=0)
    
    # Try different quantiles
    best_peaks = None
    best_pq = None
    for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        threshold = float(np.quantile(top_proj, pq))
        peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
        if module_count - 2 <= len(peaks) <= module_count + 2:
            diffs = np.diff(peaks)
            # Extended pitch range to support larger images
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 30.0)]
            if len(valid_diffs) >= 6:
                median_pitch = float(np.median(valid_diffs))
                if best_peaks is None or abs(len(peaks) - 20) < abs(len(best_peaks) - 20):
                    best_peaks = peaks
                    best_pq = pq
    
    if best_peaks is not None:
        diffs = np.diff(best_peaks)
        valid_diffs = diffs[(diffs > 8.0) & (diffs < 30.0)]
        median_pitch = float(np.median(valid_diffs))
        print(f"\n{resp_name} [pq={best_pq:.2f}]: {len(best_peaks)} peaks, pitch={median_pitch:.1f}")
        print(f"  Peaks: {best_peaks}")
        print(f"  Diffs: {diffs.tolist()}")
        
        # Try to fit progression
        top_centers = np.array(best_peaks, dtype=np.float32)
        
        # Extended fit_progression_subset with wider tolerance
        for pitch_try in np.arange(median_pitch - 3, median_pitch + 3, 0.5):
            fitted = _fit_progression_subset(top_centers.tolist(), 20, pitch_try)
            if fitted:
                print(f"  FIT SUCCESS with pitch={pitch_try:.1f}: {[f'{x:.0f}' for x in fitted]}")
                
                # Now try reconstruction!
                max_vo = max(12, int(pitch_try * 2.3))
                for vo in range(2, max_vo + 1, 2):
                    for shear in (0.0, -0.8, -1.0, -0.6, -1.2, -0.4, 0.4):
                        scores = np.zeros((20, 20), dtype=np.float32)
                        tc = np.array(fitted, dtype=np.float32)
                        ok = _fill_reconstruction_scores(scores, resp, tc, vo, pitch_try, shear)
                        if not ok:
                            continue
                        
                        for sq in (0.45, 0.50, 0.55):
                            tv = float(np.quantile(scores, sq))
                            bits = (scores >= tv).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            orient_score, oriented = _orient_bits(bits)
                            occupancy = float(bits.mean())
                            score = orient_score - abs(occupancy - 0.5) * 1.2
                            
                            if score > 3.5:
                                rendered = _render_bits(oriented)
                                decoded = _decode_pure_render(rendered)
                                if decoded:
                                    print(f"    DECODED! vo={vo}, shear={shear}, sq={sq:.2f}: {decoded}")
                                    cv2.imwrite(f"debug/cropped_success_{resp_name}.png", rendered)
                
                break  # only try first successful fit
    else:
        # Show why it failed
        top_proj = resp[:strip_height, :].mean(axis=0)
        for pq in (0.60, 0.65, 0.70):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)

print("\nDone!")
