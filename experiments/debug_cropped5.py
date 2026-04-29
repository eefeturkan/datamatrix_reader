"""Try downscaling + aggressive filtering for cropped.png"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _orient_bits, _render_bits, _decode_pure_render,
    _find_peaks_1d, _fit_progression_subset, _fill_reconstruction_scores,
    _build_reconstruction_responses, _select_top_rendered_candidates,
    _RenderedCandidate,
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Original: {w}x{h}")

# Strategy 1: Downscale to normalize pitch into 10-16 range
# Expected pitch at full size: ~22px
# Need to bring to ~12px -> scale factor = 12/22 ~= 0.55
# Or we could resize to ~250x250 which would give pitch ~12.5

for target_size in (220, 250, 280, 300):
    scale = target_size / max(h, w)
    resized = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape
    expected_pitch = rw / 20
    print(f"\nResized to {rw}x{rh}, expected pitch={expected_pitch:.1f}")
    
    # Run standard reconstruction responses
    responses = _build_reconstruction_responses(resized)
    
    strip_height = max(16, min(28, rh // 12))
    min_distance = max(8, rw // 36)
    module_count = 20
    
    for resp_name, resp in responses:
        top_proj = resp[:strip_height, :].mean(axis=0)
        
        for pq in (0.55, 0.60, 0.65, 0.70, 0.75):
            threshold = float(np.quantile(top_proj, pq))
            peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
            
            if not (module_count - 2 <= len(peaks) <= module_count + 2):
                continue
            
            diffs = np.diff(peaks)
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
            if len(valid_diffs) < 6:
                continue
            
            base_pitch = float(np.median(valid_diffs))
            if not 10.0 <= base_pitch <= 16.0:
                continue
            
            fitted = _fit_progression_subset(
                np.array(peaks, dtype=np.float32).tolist(), module_count, base_pitch
            )
            if fitted is None:
                continue
            
            top_centers = np.array(fitted, dtype=np.float32)
            print(f"  {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f}, FIT OK")
            
            # Try reconstruction
            max_vo = max(12, int(base_pitch * 2.3))
            decoded_any = False
            for vo in range(2, max_vo + 1, 2):
                for shear in (-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2):
                    for pd in (-0.4, -0.2, 0.0, 0.2, 0.4):
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
                                print(f"    DECODED! vo={vo} shear={shear} pd={pd} sq={sq:.2f}: {decoded}")
                                cv2.imwrite(
                                    f"debug/cropped_downscale_{target_size}_{resp_name}.png", 
                                    rendered
                                )
                                decoded_any = True
                                break
                        if decoded_any:
                            break
                    if decoded_any:
                        break
                if decoded_any:
                    break

# Strategy 2: Pre-filter with bilateral filter to remove texture noise while preserving dots
print("\n\n=== Strategy 2: Bilateral filtered ===")
for d_val in (7, 9, 11):
    filtered = cv2.bilateralFilter(raw, d_val, 75, 75)
    
    for target_size in (250, 280):
        scale = target_size / max(h, w)
        resized = cv2.resize(filtered, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape
        
        responses = _build_reconstruction_responses(resized)
        strip_height = max(16, min(28, rh // 12))
        min_distance = max(8, rw // 36)
        
        for resp_name, resp in responses:
            top_proj = resp[:strip_height, :].mean(axis=0)
            
            for pq in (0.55, 0.60, 0.65, 0.70, 0.75):
                threshold = float(np.quantile(top_proj, pq))
                peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
                
                if not (module_count - 2 <= len(peaks) <= module_count + 2):
                    continue
                
                diffs = np.diff(peaks)
                valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
                if len(valid_diffs) < 6:
                    continue
                
                base_pitch = float(np.median(valid_diffs))
                if not 10.0 <= base_pitch <= 16.0:
                    continue
                
                fitted = _fit_progression_subset(
                    np.array(peaks, dtype=np.float32).tolist(), module_count, base_pitch
                )
                if fitted is None:
                    continue
                
                top_centers = np.array(fitted, dtype=np.float32)
                print(f"  bilateral_d{d_val} {target_size} {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f}")
                
                max_vo = max(12, int(base_pitch * 2.3))
                decoded_any = False
                for vo in range(2, max_vo + 1, 2):
                    for shear in (-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2):
                        for pd in (-0.4, -0.2, 0.0, 0.2, 0.4):
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
                                    print(f"    DECODED! vo={vo} shear={shear} pd={pd} sq={sq:.2f}: {decoded}")
                                    cv2.imwrite(
                                        f"debug/cropped_bilateral_{d_val}_{target_size}.png", 
                                        rendered
                                    )
                                    decoded_any = True
                                    break
                            if decoded_any:
                                break
                        if decoded_any:
                            break
                    if decoded_any:
                        break

# Strategy 3: Morphological closing background subtraction + downscale
print("\n\n=== Strategy 3: Morph closing BG subtract ===")
for kernel_size in (15, 21, 31):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    subtracted = cv2.subtract(bg, raw)  # dark dots become bright
    subtracted_norm = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"debug/cropped_morphsub_{kernel_size}.png", subtracted_norm)
    
    # Also try bright dots (raw - opening)
    bg2 = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    bright_dots = cv2.subtract(raw, bg2)
    bright_dots_norm = cv2.normalize(bright_dots, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"debug/cropped_morphbright_{kernel_size}.png", bright_dots_norm)
    
    for source_name, source in [("dark", subtracted_norm), ("bright", bright_dots_norm)]:
        for target_size in (250, 280, 300):
            scale = target_size / max(h, w)
            resized = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rh, rw = resized.shape
            
            responses = _build_reconstruction_responses(resized)
            strip_height = max(16, min(28, rh // 12))
            min_distance = max(8, rw // 36)
            
            for resp_name, resp in responses:
                top_proj = resp[:strip_height, :].mean(axis=0)
                
                for pq in (0.55, 0.60, 0.65, 0.70, 0.75):
                    threshold = float(np.quantile(top_proj, pq))
                    peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)
                    
                    if not (module_count - 2 <= len(peaks) <= module_count + 2):
                        continue
                    
                    diffs = np.diff(peaks)
                    valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
                    if len(valid_diffs) < 6:
                        continue
                    
                    base_pitch = float(np.median(valid_diffs))
                    if not 10.0 <= base_pitch <= 16.0:
                        continue
                    
                    fitted = _fit_progression_subset(
                        np.array(peaks, dtype=np.float32).tolist(), module_count, base_pitch
                    )
                    if fitted is None:
                        continue
                    
                    top_centers = np.array(fitted, dtype=np.float32)
                    print(f"  morph_k{kernel_size}_{source_name} {target_size} {resp_name} pq={pq:.2f}: {len(peaks)} peaks, pitch={base_pitch:.1f}")
                    
                    max_vo = max(12, int(base_pitch * 2.3))
                    decoded_any = False
                    for vo in range(2, max_vo + 1, 2):
                        for shear in (-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4):
                            for pd in (-0.4, -0.2, 0.0, 0.2, 0.4):
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
                                        print(f"    DECODED! vo={vo} s={shear} pd={pd} sq={sq:.2f}: {decoded}")
                                        cv2.imwrite(
                                            f"debug/cropped_morph_{kernel_size}_{source_name}_{target_size}.png",
                                            rendered
                                        )
                                        decoded_any = True
                                        break
                                if decoded_any:
                                    break
                            if decoded_any:
                                break
                        if decoded_any:
                            break

print("\nDone!")
