"""Understand how test4 actually gets decoded - trace the pipeline."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _generate_roi_candidates, _build_reconstructed_candidates,
    _build_reconstruction_responses, _find_peaks_1d, _fit_progression_subset,
    _fill_reconstruction_scores, _orient_bits, _render_bits, _decode_pure_render,
    _normalize_grayscale, _FULL_RECONSTRUCTION, _FAST_RECONSTRUCTION,
    _reconstruct_datamatrix_candidates,
)

# Test4 analysis
raw4 = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\test4.png", cv2.IMREAD_GRAYSCALE)
print(f"test4: {raw4.shape[1]}x{raw4.shape[0]}, mean={raw4.mean():.0f}")

# Step 1: Check what ROI candidates are generated
rois = _generate_roi_candidates(raw4, fast_only=True)
print(f"\nFast ROI candidates: {len(rois)}")
for name, roi in rois:
    print(f"  {name}: {roi.shape[1]}x{roi.shape[0]}")

# Step 2: Check reconstruction on each ROI  
for name, roi in rois[:4]:
    print(f"\n--- Reconstruction on '{name}' ({roi.shape[1]}x{roi.shape[0]}) ---")
    candidates = _reconstruct_datamatrix_candidates(name, roi, profile=_FAST_RECONSTRUCTION)
    print(f"  Decoded candidates: {len(candidates)}")
    for c in candidates[:3]:
        print(f"    {c.stage_name}: {c.decoded_texts}")

print("\n\n" + "="*60)
print("CROPPED analysis")
print("="*60)

raw_c = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
print(f"cropped: {raw_c.shape[1]}x{raw_c.shape[0]}, mean={raw_c.mean():.0f}")

rois_c = _generate_roi_candidates(raw_c, fast_only=True)
print(f"\nFast ROI candidates: {len(rois_c)}")
for name, roi in rois_c:
    print(f"  {name}: {roi.shape[1]}x{roi.shape[0]}")

for name, roi in rois_c[:4]:
    print(f"\n--- Reconstruction on '{name}' ({roi.shape[1]}x{roi.shape[0]}) ---")
    
    # Manual check: what does reconstruction see?
    responses = _build_reconstruction_responses(roi)
    strip_height = max(16, min(28, roi.shape[0] // 12))
    min_distance = max(8, roi.shape[1] // 36)
    
    for resp_name, resp in responses[:2]:
        top_proj = resp[:strip_height, :].mean(axis=0)
        for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            if 18 <= len(peaks) <= 22:
                diffs = np.diff(peaks)
                valid = diffs[(diffs > 8.0) & (diffs < 18.0)]
                if len(valid) >= 6:
                    print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={np.median(valid):.1f}, valid_diffs={len(valid)}")

# Now try the smooth + downscale version that got FIT OK
print("\n\n--- Testing cropped_smooth_260 in detail ---")
smooth = cv2.bilateralFilter(raw_c, 11, 75, 75)
resized = cv2.resize(smooth, (260, 237), interpolation=cv2.INTER_AREA)

# Save and check responses
responses = _build_reconstruction_responses(resized)
strip_height = max(16, min(28, resized.shape[0] // 12))
min_distance = max(8, resized.shape[1] // 36)

for resp_name, resp in responses:
    cv2.imwrite(f"debug/cropped_s260_{resp_name}.png", (resp * 255).astype(np.uint8))
    
    top_proj = resp[:strip_height, :].mean(axis=0)
    for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        threshold = float(np.quantile(top_proj, pq))
        peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
        if 18 <= len(peaks) <= 22:
            diffs = np.diff(peaks)
            valid = diffs[(diffs > 8.0) & (diffs < 18.0)]
            if len(valid) >= 6:
                base_pitch = float(np.median(valid))
                fitted = _fit_progression_subset(
                    np.array(peaks, dtype=np.float32).tolist(), 20, base_pitch
                )
                if fitted:
                    print(f"\n  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f}, FIT OK")
                    print(f"    fitted: {[f'{x:.0f}' for x in fitted]}")
                    
                    top_centers = np.array(fitted, dtype=np.float32)
                    
                    # Exhaustive reconstruction
                    for vo in range(2, 50, 1):
                        for shear in np.arange(-3.0, 3.1, 0.2):
                            for pd in np.arange(-1.0, 1.1, 0.2):
                                pitch = base_pitch + pd
                                scores = np.zeros((20, 20), dtype=np.float32)
                                ok = _fill_reconstruction_scores(
                                    scores, resp, top_centers, vo, pitch, shear
                                )
                                if not ok:
                                    continue
                                
                                for sq in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
                                    tv = float(np.quantile(scores, sq))
                                    bits = (scores >= tv).astype(np.uint8)
                                    bits[0, :] = 1
                                    bits[:, 0] = 1
                                    orient_score, oriented = _orient_bits(bits)
                                    occupancy = float(bits.mean())
                                    
                                    rendered = _render_bits(oriented)
                                    decoded = _decode_pure_render(rendered)
                                    if decoded:
                                        print(f"    DECODED! vo={vo} shear={shear:.1f} pd={pd:.1f} sq={sq:.2f}: {decoded}")
                                        cv2.imwrite("debug/cropped_s260_decoded.png", rendered)
                                        # Save scores heatmap
                                        scores_vis = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                        scores_vis = cv2.resize(scores_vis, (200, 200), interpolation=cv2.INTER_NEAREST)
                                        cv2.imwrite("debug/cropped_s260_scores.png", scores_vis)
                                        break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

print("\nDone!")
