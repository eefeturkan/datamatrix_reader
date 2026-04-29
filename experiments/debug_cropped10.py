"""Compare test4 and cropped at reconstruction level - find exact parameter gap."""
import cv2
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _build_reconstruction_responses, _find_peaks_1d, _fit_progression_subset,
    _fill_reconstruction_scores, _orient_bits, _render_bits, _decode_pure_render,
)

def analyze_reconstruction(path, label):
    """Run the exact reconstruction steps and print diagnostics."""
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = raw.shape
    print(f"\n{'='*60}")
    print(f"{label}: {w}x{h}, mean={raw.mean():.0f}, std={raw.std():.0f}")
    
    responses = _build_reconstruction_responses(raw)
    strip_height = max(16, min(28, h // 12))
    min_distance = max(8, w // 36)
    
    for resp_name, resp in responses[:2]:  # mix, tophat only
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for pq in (0.65, 0.7):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            
            if not (18 <= len(peaks) <= 22):
                print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks (skip)")
                continue
            
            diffs = np.diff(peaks)
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
            if len(valid_diffs) < 6:
                # Show all diffs for debugging
                print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, valid_pitch_diffs={len(valid_diffs)}")
                print(f"    diffs: {[f'{d:.0f}' for d in diffs]}")
                continue
            
            base_pitch = float(np.median(valid_diffs))
            print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f}")
            print(f"    peaks: {peaks[:10]}...")
            print(f"    diffs: {[f'{d:.0f}' for d in diffs]}")
            
            fitted = _fit_progression_subset(
                np.array(peaks, dtype=np.float32).tolist(), 20, base_pitch
            )
            if fitted:
                print(f"    FIT OK: {[f'{x:.0f}' for x in fitted]}")
                
                # Quick reconstruction test
                top_centers = np.array(fitted, dtype=np.float32)
                decoded_count = 0
                for vo in range(2, 30, 2):
                    for shear in (-1.0, -0.8, -0.6, 0.0):
                        scores = np.zeros((20, 20), dtype=np.float32)
                        ok = _fill_reconstruction_scores(
                            scores, resp, top_centers, vo, base_pitch, shear
                        )
                        if not ok:
                            continue
                        for sq in (0.45, 0.5, 0.55):
                            tv = float(np.quantile(scores, sq))
                            bits = (scores >= tv).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            _, oriented = _orient_bits(bits)
                            rendered = _render_bits(oriented)
                            decoded = _decode_pure_render(rendered)
                            if decoded:
                                decoded_count += 1
                                if decoded_count <= 3:
                                    print(f"    DECODED vo={vo} shear={shear} sq={sq}: {decoded}")
                print(f"    Total decoded: {decoded_count}")
            else:
                print(f"    FIT FAILED")

# Analyze test4 (working reference)
t0 = time.time()
analyze_reconstruction(r"d:\DATAGUESS\datamatrix_v2\test4.png", "test4")
print(f"  (took {time.time()-t0:.1f}s)")

# Analyze cropped at original size
t0 = time.time()
analyze_reconstruction(r"d:\DATAGUESS\datamatrix_v2\cropped.png", "cropped_original")
print(f"  (took {time.time()-t0:.1f}s)")

# Analyze cropped downscaled to ~300px
t0 = time.time()
raw_cropped = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
for target_w in (260, 280, 300, 320):
    scale = target_w / raw_cropped.shape[1]
    resized = cv2.resize(raw_cropped, (target_w, int(raw_cropped.shape[0] * scale)), 
                         interpolation=cv2.INTER_AREA)
    path = f"d:\\DATAGUESS\\datamatrix_v2\\debug\\cropped_resized_{target_w}.png"
    cv2.imwrite(path, resized)
    analyze_reconstruction(path, f"cropped_{target_w}")
print(f"  (took {time.time()-t0:.1f}s)")

# Also try with bilateral smoothing before downscale
t0 = time.time()
smooth = cv2.bilateralFilter(raw_cropped, 11, 75, 75)
for target_w in (260, 280, 300):
    scale = target_w / raw_cropped.shape[1]
    resized = cv2.resize(smooth, (target_w, int(raw_cropped.shape[0] * scale)), 
                         interpolation=cv2.INTER_AREA)
    path = f"d:\\DATAGUESS\\datamatrix_v2\\debug\\cropped_smooth_resized_{target_w}.png"
    cv2.imwrite(path, resized)
    analyze_reconstruction(path, f"cropped_smooth_{target_w}")
print(f"  (took {time.time()-t0:.1f}s)")

print("\nDone!")
