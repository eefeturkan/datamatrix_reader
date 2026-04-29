"""Deep analysis of reconstruction path for cropped.png"""
import cv2
import numpy as np

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
print(f"Shape: {raw.shape}")

# Build tophat response (same as pipeline)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(raw)
sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
tophat = cv2.morphologyEx(
    sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
).astype(np.float32)
tophat_norm = cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)

# The tophat at pq=0.55-0.65 gives 20 peaks - let's see the actual peak positions and pitch
strip_height = max(16, min(28, raw.shape[0] // 12))
min_distance = max(8, raw.shape[1] // 36)

top_proj = tophat_norm[:strip_height, :].mean(axis=0)

for pq in (0.55, 0.60, 0.65):
    threshold = float(np.quantile(top_proj, pq))
    peaks = []
    for index in range(1, len(top_proj) - 1):
        current = float(top_proj[index])
        if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
            continue
        if peaks and index - peaks[-1] < min_distance:
            if current > float(top_proj[peaks[-1]]):
                peaks[-1] = index
        else:
            peaks.append(index)
    
    if len(peaks) >= 2:
        diffs = np.diff(peaks)
        valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
        print(f"\npq={pq:.2f}: {len(peaks)} peaks at positions: {peaks}")
        print(f"  Diffs: {diffs.tolist()}")
        print(f"  Valid diffs (8-18): {valid_diffs.tolist()}, count={len(valid_diffs)}")
        if len(valid_diffs) >= 6:
            median_pitch = float(np.median(valid_diffs))
            print(f"  Median pitch: {median_pitch:.2f}")
            print(f"  Pitch range check (10-16): {'PASS' if 10 <= median_pitch <= 16 else 'FAIL'}")
        else:
            # wider diffs check 
            all_diffs = diffs[(diffs > 5.0) & (diffs < 30.0)]
            print(f"  All diffs (5-30): {all_diffs.tolist()}")
            if len(all_diffs) > 0:
                print(f"  Median of wider diffs: {float(np.median(all_diffs)):.2f}")

# Also check with wider pitch tolerance
print("\n\n=== Analysis with wider pitch tolerance ===")
for pq in (0.50, 0.55, 0.60, 0.65, 0.70):
    threshold = float(np.quantile(top_proj, pq))
    peaks = []
    for index in range(1, len(top_proj) - 1):
        current = float(top_proj[index])
        if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
            continue
        if peaks and index - peaks[-1] < min_distance:
            if current > float(top_proj[peaks[-1]]):
                peaks[-1] = index
        else:
            peaks.append(index)
    
    if len(peaks) >= 2:
        diffs = np.diff(peaks)
        # Try wider pitch range
        valid_diffs = diffs[(diffs > 5.0) & (diffs < 30.0)]
        print(f"\npq={pq:.2f}: {len(peaks)} peaks")
        print(f"  Diffs: {diffs.tolist()}")
        if len(valid_diffs) >= 6:
            median_pitch = float(np.median(valid_diffs))
            print(f"  Median pitch: {median_pitch:.2f}")

# Now let's see what happens if we lower min_distance
print("\n\n=== Lower min_distance analysis ===")
for min_d in (6, 8, 10):
    for pq in (0.55, 0.60, 0.65, 0.70):
        threshold = float(np.quantile(top_proj, pq))
        peaks = []
        for index in range(1, len(top_proj) - 1):
            current = float(top_proj[index])
            if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
                continue
            if peaks and index - peaks[-1] < min_d:
                if current > float(top_proj[peaks[-1]]):
                    peaks[-1] = index
            else:
                peaks.append(index)
        
        if 18 <= len(peaks) <= 22:
            diffs = np.diff(peaks)
            valid_diffs = diffs[(diffs > 8.0) & (diffs < 18.0)]
            print(f"  min_d={min_d}, pq={pq:.2f}: {len(peaks)} peaks, valid_pitch_diffs={len(valid_diffs)}")
            if len(valid_diffs) >= 6:
                print(f"    median pitch: {np.median(valid_diffs):.2f}")

# Check how many peaks with much lower min_distance
print("\n\n=== No min_distance constraint ===")
for pq in (0.55, 0.60, 0.65, 0.70, 0.75):
    threshold = float(np.quantile(top_proj, pq))
    peaks = []
    for index in range(1, len(top_proj) - 1):
        current = float(top_proj[index])
        if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
            continue
        peaks.append(index)
    print(f"pq={pq:.2f}: {len(peaks)} raw peaks")

# Now let's check the full reconstruction pipeline path manually
print("\n\n=== Full reconstruction path ===")
# min_dist = max(8, 464 // 36) = max(8, 12) = 12
# This is the issue! For cropped.png the pitch might be around 20+ pixels
# because the image is larger than expected

# Let's compute expected pitch
# Image width = 464, 20 modules -> pitch ~ 464/20 = 23.2 pixels
print(f"Expected pitch for 20 modules in {raw.shape[1]}px wide image: {raw.shape[1]/20:.1f}px")
print(f"Expected pitch for 20 modules in {raw.shape[0]}px tall image: {raw.shape[0]/20:.1f}px")
print(f"Current pitch range check: 10.0 <= pitch <= 16.0 -> WOULD FAIL for pitch=23!")
print(f"Current valid diff range: 8.0 < diff < 18.0 -> WOULD MISS pitch=23!")

# So the problem is clear:
# 1. The image is bigger -> pitch is ~23 pixels, not 10-16
# 2. The pipeline's pitch filters (8-18 range) reject these wider-pitched dots
# 3. The min_distance of 12 might also be too restrictive

# Verify with correct pitch range
print("\n=== With corrected pitch range ===")
for pq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
    threshold = float(np.quantile(top_proj, pq))
    peaks = []
    for index in range(1, len(top_proj) - 1):
        current = float(top_proj[index])
        if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
            continue
        if peaks and index - peaks[-1] < 12:
            if current > float(top_proj[peaks[-1]]):
                peaks[-1] = index
        else:
            peaks.append(index)
    
    if len(peaks) >= 2:
        diffs = np.diff(peaks)
        # Extended pitch range
        valid_diffs = diffs[(diffs > 8.0) & (diffs < 35.0)]
        if 18 <= len(peaks) <= 22:
            median_pitch = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else 0
            print(f"  pq={pq:.2f}: {len(peaks)} peaks, valid_diffs(8-35)={len(valid_diffs)}, median_pitch={median_pitch:.1f}")
            print(f"    diffs: {diffs.tolist()}")

# Now check if reconstruction would work with wider pitch
print("\n=== Reconstruction grid test with wider pitch ===")
# Using pq=0.60 which gives 20 peaks
pq = 0.60
threshold = float(np.quantile(top_proj, pq))
peaks = []
for index in range(1, len(top_proj) - 1):
    current = float(top_proj[index])
    if current < threshold or current < top_proj[index - 1] or current <= top_proj[index + 1]:
        continue
    if peaks and index - peaks[-1] < 12:
        if current > float(top_proj[peaks[-1]]):
            peaks[-1] = index
    else:
        peaks.append(index)

print(f"Peaks at pq=0.60: {peaks}")
print(f"Count: {len(peaks)}")
if len(peaks) >= 2:
    diffs = np.diff(peaks)
    print(f"Diffs: {diffs.tolist()}")
    valid_extended = diffs[(diffs > 12.0) & (diffs < 35.0)]
    if len(valid_extended) >= 6:
        pitch = float(np.median(valid_extended))
        print(f"Median pitch (extended): {pitch:.1f}")
        
        # Try grid fitting
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
        from datamatrix_reader.pipeline import _fit_progression_subset
        
        top_centers = np.array(peaks, dtype=np.float32)
        fitted = _fit_progression_subset(top_centers.tolist(), 20, pitch)
        if fitted:
            print(f"  Fitted progression: {[f'{x:.1f}' for x in fitted]}")
        else:
            print(f"  Fit FAILED")
