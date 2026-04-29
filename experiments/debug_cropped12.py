"""Final approach: Create a clean binary from dot positions using iterative
morphological filtering with size-based filtering to remove non-dot artifacts."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

# Step 1: Bilateral + CLAHE to clean up
smooth = cv2.bilateralFilter(raw, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)

# Step 2: Tophat to extract bright features (dots have bright centers on this image)
for tophat_size in (9, 11, 13, 15, 17):
    tophat = cv2.morphologyEx(
        clahe, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_size, tophat_size))
    )
    
    # Step 3: Threshold to get binary
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 4: Filter by component size - keep only dot-sized components
    # Expected dot diameter: image_width / 20 modules * ~0.6 fill = ~14px diameter
    # So dot area ~ pi * 7^2 = ~154 pixels
    expected_dot_area = (w / 20 * 0.6) ** 2 * 3.14
    min_dot_area = expected_dot_area * 0.1
    max_dot_area = expected_dot_area * 3.0
    
    cc_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    
    # Create filtered binary
    filtered = np.zeros_like(binary)
    dot_centers = []
    for i in range(1, cc_count):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        
        if min_dot_area <= area <= max_dot_area and aspect > 0.3:
            filtered[labels == i] = 255
            dot_centers.append(centroids[i])
    
    dot_count = len(dot_centers)
    cv2.imwrite(f"debug/cropped_filtered_th{tophat_size}.png", filtered)
    
    if dot_count < 100:
        continue
    
    print(f"\n  tophat_size={tophat_size}: {dot_count} dots (area range: {min_dot_area:.0f}-{max_dot_area:.0f})")
    
    # Step 5: Create a cleaner binary by drawing filled circles at dot centers
    # Estimate dot radius from the filtered components
    radii = []
    for i in range(1, cc_count):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_dot_area <= area <= max_dot_area:
            radii.append(np.sqrt(area / 3.14))
    median_radius = np.median(radii) if radii else 5
    
    # Draw dots as filled circles
    clean_binary = np.zeros_like(raw)
    for cx, cy in dot_centers:
        cv2.circle(clean_binary, (int(cx), int(cy)), int(median_radius), 255, -1)
    
    cv2.imwrite(f"debug/cropped_clean_th{tophat_size}.png", clean_binary)
    
    # Try to decode the clean binary and filtered versions
    for name, img in [("filtered", filtered), ("clean", clean_binary)]:
        for scale in (1, 2, 3):
            scaled = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, 
                                                        interpolation=cv2.INTER_NEAREST)
            for rot in (0, 90, 180, 270):
                rotated = scaled
                if rot == 90: rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180: rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                elif rot == 270: rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                for inv in (False, True):
                    test_img = 255 - rotated if inv else rotated
                    try:
                        results = zxingcpp.read_barcodes(
                            test_img,
                            formats=zxingcpp.BarcodeFormat.DataMatrix,
                            try_rotate=True, try_downscale=True,
                            text_mode=zxingcpp.TextMode.Plain,
                        )
                        for r in (results or []):
                            text = (getattr(r, "text", "") or "").strip()
                            if text:
                                inv_s = "_inv" if inv else ""
                                print(f"    ZXING DECODED! th{tophat_size}_{name}_x{scale}_r{rot}{inv_s}: {text}")
                                cv2.imwrite(f"debug/cropped_decoded_th{tophat_size}_{name}.png", test_img)
                    except:
                        pass

# Step 6: Also try blackhat (dark features) 
print("\n=== Blackhat approach ===")
for bh_size in (9, 13, 17, 21):
    blackhat = cv2.morphologyEx(
        clahe, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
    )
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cc_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros_like(binary)
    dot_count = 0
    for i in range(1, cc_count):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh_val = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(bw, bh_val) / max(bw, bh_val) if max(bw, bh_val) > 0 else 0
        if min_dot_area <= area <= max_dot_area and aspect > 0.3:
            filtered[labels == i] = 255
            dot_count += 1
    
    cv2.imwrite(f"debug/cropped_bh_filtered_{bh_size}.png", filtered)
    
    if dot_count >= 100:
        print(f"  bh_size={bh_size}: {dot_count} dots")
        for scale in (2, 3):
            scaled = cv2.resize(filtered, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            for rot in (0, 90, 180, 270):
                rotated = scaled
                if rot == 90: rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180: rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                elif rot == 270: rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                for inv in (False, True):
                    test_img = 255 - rotated if inv else rotated
                    try:
                        results = zxingcpp.read_barcodes(
                            test_img, formats=zxingcpp.BarcodeFormat.DataMatrix,
                            try_rotate=True, try_downscale=True,
                            text_mode=zxingcpp.TextMode.Plain,
                        )
                        for r in (results or []):
                            text = (getattr(r, "text", "") or "").strip()
                            if text:
                                inv_s = "_inv" if inv else ""
                                print(f"    ZXING! bh{bh_size}_x{scale}_r{rot}{inv_s}: {text}")
                    except:
                        pass

# Step 7: Dilated dot version (fill gaps between nearby dot components)
print("\n=== Dilated approach ===")
for tophat_size in (11, 15):
    tophat = cv2.morphologyEx(
        clahe, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_size, tophat_size))
    )
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilate to connect nearby components
    dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    # Then erode back
    eroded = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Filter by size
    cc_count, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, 8)
    filtered = np.zeros_like(binary)
    for i in range(1, cc_count):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh_val = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(bw, bh_val) / max(bw, bh_val) if max(bw, bh_val) > 0 else 0
        if min_dot_area * 0.5 <= area <= max_dot_area * 2.0 and aspect > 0.25:
            filtered[labels == i] = 255
    
    cv2.imwrite(f"debug/cropped_dilated_th{tophat_size}.png", filtered)
    
    for scale in (2, 3):
        scaled = cv2.resize(filtered, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for inv in (False, True):
            test_img = 255 - scaled if inv else scaled
            try:
                results = zxingcpp.read_barcodes(
                    test_img, formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True, try_downscale=True,
                    text_mode=zxingcpp.TextMode.Plain,
                )
                for r in (results or []):
                    text = (getattr(r, "text", "") or "").strip()
                    if text:
                        inv_s = "_inv" if inv else ""
                        print(f"    ZXING! dil_th{tophat_size}_x{scale}{inv_s}: {text}")
            except:
                pass
            
            try:
                results = dmtx_decode(test_img, timeout=500, max_count=3, corrections=10)
                for r in (results or []):
                    text = r.data.decode("utf-8", "replace").strip()
                    if text:
                        inv_s = "_inv" if inv else ""
                        print(f"    DMTX! dil_th{tophat_size}_x{scale}{inv_s}: {text}")
            except:
                pass

print("\nDone!")
