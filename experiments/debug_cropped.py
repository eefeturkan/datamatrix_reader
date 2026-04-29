"""Debug script to understand why cropped.png fails in reconstruction."""
import cv2
import numpy as np
from pathlib import Path

img_path = Path(r"d:\DATAGUESS\datamatrix_v2\cropped.png")
raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {raw.shape}")
print(f"Mean intensity: {raw.mean():.1f}, Std: {raw.std():.1f}")
print(f"Min: {raw.min()}, Max: {raw.max()}")

# Check what _build_reconstruction_responses produces
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(raw)
sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
tophat = cv2.morphologyEx(
    sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
).astype(np.float32)
blackhat = cv2.morphologyEx(
    sharpen, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
).astype(np.float32)

mix = cv2.normalize(0.65 * tophat + 0.35 * blackhat, None, 0, 1, cv2.NORM_MINMAX)
tophat_norm = cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)

print(f"\nCLAHE mean: {clahe.mean():.1f}, std: {clahe.std():.1f}")
print(f"Sharpen mean: {sharpen.mean():.1f}, std: {sharpen.std():.1f}")
print(f"Tophat mean: {tophat.mean():.1f}, std: {tophat.std():.1f}, max: {tophat.max():.1f}")
print(f"Blackhat mean: {blackhat.mean():.1f}, std: {blackhat.std():.1f}, max: {blackhat.max():.1f}")

# Save debug images
cv2.imwrite("debug/cropped_clahe.png", clahe)
cv2.imwrite("debug/cropped_sharpen.png", sharpen)
cv2.imwrite("debug/cropped_tophat.png", (tophat_norm * 255).astype(np.uint8))
cv2.imwrite("debug/cropped_mix.png", (mix * 255).astype(np.uint8))

# Check reconstruction peak detection
strip_height = max(16, min(28, raw.shape[0] // 12))
print(f"\nstrip_height: {strip_height}")

for resp_name, resp in [("mix", mix), ("tophat", tophat_norm)]:
    top_proj = resp[:strip_height, :].mean(axis=0)
    print(f"\n--- {resp_name} ---")
    print(f"Top projection: mean={top_proj.mean():.3f}, max={top_proj.max():.3f}, std={top_proj.std():.3f}")
    
    min_distance = max(8, raw.shape[1] // 36)
    print(f"min_distance: {min_distance}")
    
    for pq in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        threshold = float(np.quantile(top_proj, pq))
        # Find peaks
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
        print(f"  pq={pq:.2f}: threshold={threshold:.3f}, peaks={len(peaks)}")

# Check contour-based ROI detection
print("\n=== Contour-based ROI detection ===")
height, width = raw.shape
for invert in (False, True):
    base = 255 - raw if invert else raw
    blurred = cv2.GaussianBlur(base, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3,
    )
    closed = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    accepted = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < 0.08 * width * height or area > 0.98 * width * height:
            continue
        ratio = w / max(h, 1)
        if ratio < 0.65 or ratio > 1.35:
            continue
        accepted += 1
    print(f"  invert={invert}: total_contours={len(contours)}, accepted={accepted}")

# Check connected component detection (row localization)
print("\n=== Connected Component Analysis ===")
for resp_name, resp in [("mix", mix), ("tophat", tophat_norm)]:
    for quantile in (0.85, 0.88, 0.9, 0.92):
        threshold_value = np.quantile(resp, quantile)
        binary = (resp >= threshold_value).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        
        valid_points = 0
        for index in range(1, component_count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if 3 <= area <= 120:
                valid_points += 1
        
        print(f"  {resp_name} q={quantile:.2f}: components={component_count}, valid_points={valid_points}")

# Now let's try with inverted image 
print("\n=== Inverted image analysis ===")
inverted = 255 - raw
clahe_inv = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(inverted)
sharpen_inv = cv2.addWeighted(clahe_inv, 1.8, cv2.GaussianBlur(clahe_inv, (0, 0), 1.0), -0.8, 0)
tophat_inv = cv2.morphologyEx(
    sharpen_inv, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
).astype(np.float32)
blackhat_inv = cv2.morphologyEx(
    sharpen_inv, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
).astype(np.float32)

mix_inv = cv2.normalize(0.65 * tophat_inv + 0.35 * blackhat_inv, None, 0, 1, cv2.NORM_MINMAX)
tophat_inv_norm = cv2.normalize(tophat_inv, None, 0, 1, cv2.NORM_MINMAX)
print(f"Inverted Tophat mean: {tophat_inv.mean():.1f}, std: {tophat_inv.std():.1f}, max: {tophat_inv.max():.1f}")
print(f"Inverted Blackhat mean: {blackhat_inv.mean():.1f}, std: {blackhat_inv.std():.1f}, max: {blackhat_inv.max():.1f}")

cv2.imwrite("debug/cropped_inv_tophat.png", (tophat_inv_norm * 255).astype(np.uint8))
cv2.imwrite("debug/cropped_inv_mix.png", (mix_inv * 255).astype(np.uint8))

for resp_name, resp in [("inv_mix", mix_inv), ("inv_tophat", tophat_inv_norm)]:
    top_proj = resp[:strip_height, :].mean(axis=0)
    min_distance = max(8, raw.shape[1] // 36)
    
    for pq in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
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
        print(f"  {resp_name} pq={pq:.2f}: threshold={threshold:.3f}, peaks={len(peaks)}")

    # CC analysis
    for quantile in (0.85, 0.88, 0.9, 0.92):
        threshold_value = np.quantile(resp, quantile)
        binary = (resp >= threshold_value).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        
        valid_points = 0
        for index in range(1, component_count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if 3 <= area <= 120:
                valid_points += 1
        print(f"  {resp_name} CC q={quantile:.2f}: components={component_count}, valid_points={valid_points}")

# Try different preprocessing for bright background
print("\n=== Bright-background specific preprocessing ===")
# The dots in cropped.png appear as raised/bright bumps on a bright background
# This means we need to detect local bright peaks on a bright background

# Try using a local contrast enhancement approach
# Normalize using local mean subtraction
kernel_size = 51
local_mean = cv2.blur(raw, (kernel_size, kernel_size))
local_contrast = cv2.subtract(raw, local_mean)
local_contrast_enhanced = cv2.normalize(local_contrast, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("debug/cropped_local_contrast.png", local_contrast_enhanced)

# Try blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 15
params.maxArea = 500
params.filterByCircularity = True
params.minCircularity = 0.3
params.filterByConvexity = False
params.filterByInertia = False

# For bright blobs on bright background - we need to invert first or detect dark blobs on inverted
for name, img in [("raw", raw), ("clahe", clahe), ("inverted", inverted), ("clahe_inv", clahe_inv)]:
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    print(f"  Blob detector on {name}: {len(keypoints)} blobs found")

# DoG (Difference of Gaussians) approach
print("\n=== DoG approach ===")
for sigma1, sigma2 in [(1, 3), (1, 5), (2, 5), (2, 7), (3, 7)]:
    g1 = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), sigma1)
    g2 = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), sigma2)
    dog = g1 - g2
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Count peaks
    thresh = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(thresh, 8)
    valid = sum(1 for i in range(1, cc) if 3 <= stats_cc[i, cv2.CC_STAT_AREA] <= 200)
    print(f"  DoG({sigma1},{sigma2}): valid_components={valid}")
    
    if sigma1 == 2 and sigma2 == 5:
        cv2.imwrite("debug/cropped_dog.png", dog_norm)
        cv2.imwrite("debug/cropped_dog_thresh.png", thresh)

print("\nDone!")
