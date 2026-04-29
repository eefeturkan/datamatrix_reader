"""Comprehensive approach: widen pitch tolerance + enhanced responses + full search"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _orient_bits, _render_bits, _decode_pure_render,
    _find_peaks_1d, _fit_progression_subset, _fill_reconstruction_scores,
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape

# The problem: cropped.png has dot-peen marks on a bright metallic surface.
# The dots appear as both highlights AND shadows depending on lighting angle.
# The current pipeline tophat finds the specular highlights on dot edges,
# but these highlight positions don't align with dot CENTERS.

# New approach: Create a response map that captures DOT CENTERS, not edges
# Step 1: Reduce texture noise with morphological operations
# Step 2: Create a dot-center response using matched filtering

def create_dot_center_response(roi, dot_radius=5):
    """Create a response map that highlights dot centers using matched filtering."""
    # Bilateral filter to smooth texture while preserving dot edges
    smoothed = cv2.bilateralFilter(roi, 9, 75, 75)
    
    # Create responses for both bright and dark dots
    responses = []
    
    # Approach 1: Background subtraction with median filter (very good for regular dots)
    for ksize in (21, 31):
        bg = cv2.medianBlur(smoothed, ksize)
        # Bright dots
        bright = cv2.subtract(smoothed, bg)
        bright_norm = cv2.normalize(bright, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        responses.append((f"median_bright_{ksize}", bright_norm))
        # Dark dots
        dark = cv2.subtract(bg, smoothed)
        dark_norm = cv2.normalize(dark, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        responses.append((f"median_dark_{ksize}", dark_norm))
    
    # Approach 2: Matched filter with circular kernel
    for radius in (4, 5, 6, 7):
        kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
        kernel /= kernel.sum()
        # Surround with negative ring
        outer = np.zeros((2*radius+3, 2*radius+3), dtype=np.float32)
        cv2.circle(outer, (radius+1, radius+1), radius+1, 1.0, -1)
        inner = np.zeros((2*radius+3, 2*radius+3), dtype=np.float32)
        cv2.circle(inner, (radius+1, radius+1), max(1, radius-1), 1.0, -1)
        ring = outer - inner
        ring_sum = ring.sum()
        if ring_sum > 0:
            ring /= ring_sum
        
        matched = cv2.filter2D(smoothed.astype(np.float32), cv2.CV_32F, inner / max(inner.sum(), 1)) - \
                  cv2.filter2D(smoothed.astype(np.float32), cv2.CV_32F, ring)
        
        for name_suffix, signal in [("pos", np.clip(matched, 0, None)), ("neg", np.clip(-matched, 0, None))]:
            norm = cv2.normalize(signal, None, 0, 1, cv2.NORM_MINMAX)
            if norm is not None and np.std(norm) > 0.05:
                responses.append((f"matched_r{radius}_{name_suffix}", norm))
    
    # Standard tophat/blackhat on smoothed
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(smoothed)
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    blackhat = cv2.morphologyEx(
        sharpen, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    responses.append(("smooth_tophat", cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)))
    responses.append(("smooth_blackhat", cv2.normalize(blackhat, None, 0, 1, cv2.NORM_MINMAX)))
    responses.append(("smooth_mix", cv2.normalize(0.65*tophat + 0.35*blackhat, None, 0, 1, cv2.NORM_MINMAX)))
    
    return responses

responses = create_dot_center_response(raw)

# Save all response maps
for name, resp in responses:
    cv2.imwrite(f"debug/cropped_resp_{name}.png", (resp * 255).astype(np.uint8))

# Extended pitch range: allow 8-28 instead of 10-16
# Extended peak diff range: allow 6-30 instead of 8-18
module_count = 20
strip_height = max(16, min(28, h // 12))

print("=== Extended pitch reconstruction ===")

for resp_name, resp in responses:
    top_proj = resp[:strip_height, :].mean(axis=0)
    
    decoded_any = False
    for min_dist_ratio in (24, 30, 36):
        if decoded_any:
            break
        min_distance = max(6, w // min_dist_ratio)
        
        for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            
            if not (module_count - 2 <= len(peaks) <= module_count + 2):
                continue
            
            diffs = np.diff(peaks)
            # Extended range
            valid_diffs = diffs[(diffs > 6.0) & (diffs < 30.0)]
            if len(valid_diffs) < 6:
                continue
            
            base_pitch = float(np.median(valid_diffs))
            if not 8.0 <= base_pitch <= 28.0:
                continue
            
            # Try fit with extended tolerance
            fitted = _fit_progression_subset(
                np.array(peaks, dtype=np.float32).tolist(), module_count, base_pitch
            )
            if fitted is None:
                # Try wider tolerance fit
                values = sorted(np.array(peaks, dtype=np.float32).tolist())
                best_score = -1
                best_progression = None
                for start in values:
                    for pitch_delta in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
                        pitch_try = base_pitch + pitch_delta
                        expected = [start + i * pitch_try for i in range(module_count)]
                        matched = 0
                        progression = []
                        for center in expected:
                            nearest = min(values, key=lambda v: abs(v - center))
                            if abs(nearest - center) <= pitch_try * 0.35:  # wider tolerance
                                matched += 1
                                progression.append(nearest)
                            else:
                                progression.append(center)
                        if matched > best_score:
                            best_score = matched
                            best_progression = progression
                
                if best_score < module_count - 5:
                    continue
                fitted = best_progression
            
            top_centers = np.array(fitted, dtype=np.float32)
            
            # Extended vertical search
            max_vo = max(12, int(base_pitch * 2.5))
            decoded_any = False
            
            for vo in range(2, max_vo + 1, 2):
                for shear in np.arange(-2.0, 2.1, 0.4):
                    for pd in np.arange(-1.0, 1.1, 0.2):
                        pitch = base_pitch + pd
                        scores = np.zeros((20, 20), dtype=np.float32)
                        ok = _fill_reconstruction_scores(
                            scores, resp, top_centers, vo, pitch, shear
                        )
                        if not ok:
                            continue
                        
                        for sq in (0.40, 0.45, 0.50, 0.55, 0.60):
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
                                print(f"  DECODED! {resp_name} md={min_distance} pq={pq:.2f} "
                                      f"pitch={base_pitch:.1f} vo={vo} shear={shear:.1f} "
                                      f"pd={pd:.1f} sq={sq:.2f}: {decoded}")
                                cv2.imwrite(
                                    f"debug/cropped_ext_{resp_name}.png", rendered
                                )
                                decoded_any = True
                                break
                        if decoded_any:
                            break
                    if decoded_any:
                        break
                if decoded_any:
                    break
            
            if decoded_any:
                break
        if decoded_any:
            break

print("\nDone!")
