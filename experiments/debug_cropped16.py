"""Trace test4's full reconstruction path to understand exactly what makes it work."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _generate_roi_candidates, _build_reconstructed_candidates,
    _build_reconstruction_responses, _find_peaks_1d, _fit_progression_subset,
    _fill_reconstruction_scores, _orient_bits, _render_bits, _decode_pure_render,
    _reconstruct_datamatrix_candidates, _FULL_RECONSTRUCTION, _FAST_RECONSTRUCTION,
)

# Test4
raw4 = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\test4.png", cv2.IMREAD_GRAYSCALE)
print(f"test4: {raw4.shape[1]}x{raw4.shape[0]}")

# Full reconstruction (not fast) - this is what actually works for test4
rois = _generate_roi_candidates(raw4, fast_only=False)
print(f"\nFull ROI candidates ({len(rois)}):")
for name, roi in rois:
    print(f"  {name}: {roi.shape[1]}x{roi.shape[0]}")

# Check each ROI with FULL profile
for name, roi in rois:
    print(f"\n--- {name} ({roi.shape[1]}x{roi.shape[0]}) ---")
    responses = _build_reconstruction_responses(roi)
    
    strip_height = max(16, min(28, roi.shape[0] // 12))
    min_distance = max(8, roi.shape[1] // 36)
    
    found_any = False
    for resp_name, resp in responses:
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for pq in _FULL_RECONSTRUCTION.projection_quantiles:
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
            
            if not found_any:
                print(f"  FIRST FIT: {resp_name} pq={pq:.2f}, {len(peaks)} peaks, pitch={base_pitch:.1f}")
                print(f"    fitted: {[f'{x:.0f}' for x in fitted]}")
                print(f"    diffs: {[f'{d:.0f}' for d in np.diff(fitted)]}")
                found_any = True
    
    if found_any:
        # Now run actual reconstruction
        candidates = _reconstruct_datamatrix_candidates(name, roi, profile=_FULL_RECONSTRUCTION)
        if candidates:
            print(f"  DECODED CANDIDATES: {len(candidates)}")
            for c in candidates[:2]:
                print(f"    {c.stage_name}: {c.decoded_texts}")
            break
        else:
            print(f"  FIT OK but no decode")

# Now do the same for cropped
print("\n\n" + "="*60)
raw_c = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
print(f"cropped: {raw_c.shape[1]}x{raw_c.shape[0]}")

rois_c = _generate_roi_candidates(raw_c, fast_only=False)
print(f"\nFull ROI candidates ({len(rois_c)}):")
for name, roi in rois_c:
    print(f"  {name}: {roi.shape[1]}x{roi.shape[0]}")

# Check if any ROI gets good peaks with wider parameters
for name, roi in rois_c:
    print(f"\n--- {name} ({roi.shape[1]}x{roi.shape[0]}) ---")
    responses = _build_reconstruction_responses(roi)
    
    strip_height = max(16, min(28, roi.shape[0] // 12))
    min_distance = max(8, roi.shape[1] // 36)
    
    for resp_name, resp in responses:
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            
            if not (18 <= len(peaks) <= 22):
                continue
            
            diffs = np.diff(peaks)
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
            if len(valid_diffs) < 6:
                # Show for debugging
                all_valid = diffs[(diffs > 5.0) & (diffs < 35.0)]
                if len(peaks) == 20:
                    print(f"  {resp_name} pq={pq:.2f}: 20 peaks, valid(8-18)={len(valid_diffs)}, valid(5-35)={len(all_valid)}")
                    print(f"    median(all_valid)={np.median(all_valid):.1f}" if len(all_valid) > 0 else "")
                continue
            
            base_pitch = float(np.median(valid_diffs))
            if not 10.0 <= base_pitch <= 16.0:
                print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f} OUT OF RANGE")
                continue
            
            fitted = _fit_progression_subset(
                np.array(peaks, dtype=np.float32).tolist(), 20, base_pitch
            )
            if fitted is None:
                print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f} FIT FAILED")
            else:
                print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f} FIT OK")
                print(f"    fitted diffs: {[f'{d:.0f}' for d in np.diff(fitted)]}")

print("\nDone!")
