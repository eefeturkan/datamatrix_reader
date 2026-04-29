"""Find the exact difference: why do cropped.png's 20 peaks fail _fit_progression_subset
while test4's 20 peaks succeed? And what preprocessing would fix it?"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _build_reconstruction_responses, _find_peaks_1d, _fit_progression_subset,
    _fill_reconstruction_scores, _orient_bits, _render_bits, _decode_pure_render,
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape

# The "full" ROI with tophat pq=0.65 gives 20 peaks with pitch=14 but FIT FAILS.
# Let's see exactly why.

responses = _build_reconstruction_responses(raw)
tophat_resp = None
for name, resp in responses:
    if name == "tophat":
        tophat_resp = resp
        break

strip_height = max(16, min(28, h // 12))
min_distance = max(8, w // 36)

top_proj = tophat_resp[:strip_height, :].mean(axis=0)
threshold = float(np.quantile(top_proj, 0.65))
peaks = _find_peaks_1d(top_proj, min_dist=min_distance, threshold=threshold)

print(f"20 peaks: {peaks}")
print(f"Diffs: {np.diff(peaks).tolist()}")

# _fit_progression_subset needs at least module_count - 3 = 17 matches within 4.5px
# Let's try all starts and see best match count
values = sorted(float(p) for p in peaks)
base_pitch = 14.0

print(f"\n=== Fit analysis with base_pitch={base_pitch:.1f} ===")
for start in values:
    for pd in (-1.0, -0.5, 0.0, 0.5, 1.0):
        pitch = base_pitch + pd
        expected = [start + i * pitch for i in range(20)]
        matched = 0
        for e in expected:
            nearest = min(values, key=lambda v: abs(v - e))
            if abs(nearest - e) <= 4.5:
                matched += 1
        if matched >= 14:
            print(f"  start={start:.0f} pitch={pitch:.1f}: {matched}/20 matched")

# Now try with more tolerance
print(f"\n=== With tolerance 7.0 ===")
for start in values:
    for pd in (-1.0, -0.5, 0.0, 0.5, 1.0):
        pitch = base_pitch + pd
        expected = [start + i * pitch for i in range(20)]
        matched = 0
        for e in expected:
            nearest = min(values, key=lambda v: abs(v - e))
            if abs(nearest - e) <= 7.0:
                matched += 1
        if matched >= 17:
            print(f"  start={start:.0f} pitch={pitch:.1f}: {matched}/20 matched")

# The issue: peaks at positions [9, 31, 45, 67, 98, 122, 156, 190, 206, 218, 231, 248, 267, 281, 306, 340, 353, 391, 417, 453]
# Expected with pitch 22.3: 9, 31.3, 53.6, 75.9, 98.2, 120.5, 142.8, 165.1, 187.4, 209.7, 232, 254.3, 276.6, 298.9, 321.2, 343.5, 365.8, 388.1, 410.4, 432.7
# Real pitch is ~22-23px, not 14!

print(f"\n\n=== Correct pitch analysis ===")
# The real pitch for 20 dots in 464px wide image
real_pitch = (453 - 9) / 19  # from first to last peak
print(f"True pitch estimate: {real_pitch:.1f}")

for base_p in np.arange(20.0, 26.0, 0.5):
    for start in values[:5]:
        expected = [start + i * base_p for i in range(20)]
        matched = 0
        matched_pairs = []
        for i, e in enumerate(expected):
            nearest = min(values, key=lambda v: abs(v - e))
            if abs(nearest - e) <= base_p * 0.25:
                matched += 1
                matched_pairs.append((i, nearest, e))
        if matched >= 12:
            print(f"  start={start:.0f} pitch={base_p:.1f}: {matched}/20 matched")
            if matched >= 15:
                # Show details
                for idx, actual, expected_val in matched_pairs:
                    print(f"    [{idx}] expected={expected_val:.1f} actual={actual:.0f} diff={actual-expected_val:.1f}")

# Now test: if we use the correct pitch, can we reconstruct?
print(f"\n=== Reconstruction with correct pitch ===")
real_p = 23.4  # from above analysis
fitted = [9 + i * real_p for i in range(20)]
# Snap to nearest peaks
snapped = []
for e in fitted:
    nearest = min(values, key=lambda v: abs(v - e))
    if abs(nearest - e) <= real_p * 0.25:
        snapped.append(nearest)
    else:
        snapped.append(e)
print(f"Snapped grid: {[f'{x:.0f}' for x in snapped]}")

top_centers = np.array(snapped, dtype=np.float32)

decoded_count = 0
for resp_name, resp in responses:
    for vo in range(2, 60, 2):
        for shear in np.arange(-3.0, 3.1, 0.5):
            for pd in np.arange(-1.5, 1.6, 0.3):
                pitch = real_p + pd
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
                    
                    rendered = _render_bits(oriented)
                    decoded = _decode_pure_render(rendered)
                    if decoded:
                        decoded_count += 1
                        if decoded_count <= 5:
                            print(f"  DECODED! {resp_name} vo={vo} shear={shear:.1f} pd={pd:.1f} sq={sq:.2f}: {decoded}")
                            cv2.imwrite(f"debug/cropped_realp_decoded_{decoded_count}.png", rendered)
                            # Save scores
                            scores_vis = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            scores_big = cv2.resize(scores_vis, (200, 200), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(f"debug/cropped_realp_scores_{decoded_count}.png", scores_big)

print(f"\nTotal decoded: {decoded_count}")
print("Done!")
