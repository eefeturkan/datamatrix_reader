"""Fast targeted approach: create clean binary images and try decoders directly."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _decode_pure_render, _orient_bits, _render_bits

import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

def try_decode_zxing(image, label=""):
    """Try zxing decode on an image."""
    for try_img in [image, 255 - image]:
        try:
            results = zxingcpp.read_barcodes(
                try_img,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=True,
                text_mode=zxingcpp.TextMode.Plain,
            )
            for r in (results or []):
                text = (getattr(r, "text", "") or "").strip()
                if text:
                    print(f"  ZXING [{label}]: {text}")
                    return text
        except:
            pass
    return None

def try_decode_dmtx(image, label=""):
    """Try pylibdmtx decode on an image."""
    for try_img in [image, 255 - image]:
        try:
            results = dmtx_decode(try_img, timeout=500, max_count=3, corrections=10)
            for r in (results or []):
                text = r.data.decode("utf-8", "replace").strip()
                if text:
                    print(f"  DMTX [{label}]: {text}")
                    return text
        except:
            pass
    return None

# ============================================================
# Strategy 1: Clean binary via morphological background removal
# ============================================================
print("\n=== Strategy 1: Morphological BG removal ===")
smoothed = cv2.bilateralFilter(raw, 11, 75, 75)

for kernel_size in (15, 21, 31, 41):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # White tophat (bright dots on dark bg)
    wth = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, kernel)
    # Black tophat (dark dots on bright bg) 
    bth = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel)
    
    for name, resp in [("wth", wth), ("bth", bth)]:
        # Otsu threshold
        _, binary = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try multiple scales
        for scale in (2, 3, 4):
            scaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            for rot in (0, 90, 180, 270):
                rotated = scaled
                if rot == 90:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                elif rot == 270:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                label = f"morph_k{kernel_size}_{name}_x{scale}_r{rot}"
                result = try_decode_zxing(rotated, label)
                if result:
                    cv2.imwrite(f"debug/cropped_success_{label}.png", rotated)

# ============================================================
# Strategy 2: Adaptive threshold variants
# ============================================================
print("\n=== Strategy 2: Adaptive threshold ===")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(raw)
sharpen = cv2.addWeighted(clahe, 2.0, cv2.GaussianBlur(clahe, (0, 0), 1.5), -1.0, 0)

for block_size in (21, 31, 41, 51):
    for c_val in (2, 5, 8, 12):
        for src_name, src in [("clahe", clahe), ("sharp", sharpen), ("smooth", smoothed)]:
            adapt = cv2.adaptiveThreshold(
                src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
            )
            
            for scale in (2, 3):
                scaled = cv2.resize(adapt, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                for rot in (0, 90, 180, 270):
                    rotated = scaled
                    if rot == 90:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                    elif rot == 270:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    label = f"adapt_{src_name}_b{block_size}_c{c_val}_x{scale}_r{rot}"
                    result = try_decode_zxing(rotated, label)
                    if result:
                        cv2.imwrite(f"debug/cropped_success_{label}.png", rotated)

# ============================================================
# Strategy 3: Combined approach - morphological cleaning + threshold
# ============================================================
print("\n=== Strategy 3: Combined morph + threshold ===")
for kernel_size in (15, 21, 31):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Background subtraction
    bg = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    fg = cv2.subtract(bg, smoothed)
    fg_norm = cv2.normalize(fg, None, 0, 255, cv2.NORM_MINMAX)
    
    bg2 = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    fg2 = cv2.subtract(smoothed, bg2)
    fg2_norm = cv2.normalize(fg2, None, 0, 255, cv2.NORM_MINMAX)
    
    for name, img in [("dark_dots", fg_norm), ("bright_dots", fg2_norm)]:
        # Threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        clean = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        clean = cv2.morphologyEx(
            clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        
        for scale in (2, 3, 4):
            scaled = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            for rot in (0, 90, 180, 270):
                rotated = scaled
                if rot == 90:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                elif rot == 270:
                    rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                label = f"comb_k{kernel_size}_{name}_x{scale}_r{rot}"
                result = try_decode_zxing(rotated, label)
                if result:
                    cv2.imwrite(f"debug/cropped_success_{label}.png", rotated)

# ============================================================
# Strategy 4: Try pylibdmtx with longer timeout on best preprocessed
# ============================================================
print("\n=== Strategy 4: pylibdmtx ===")
for kernel_size in (21, 31):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    wth = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, kernel)
    bth = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel)
    
    for name, resp in [("wth", wth), ("bth", bth)]:
        _, binary = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for scale in (3, 4):
            scaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            label = f"dmtx_k{kernel_size}_{name}_x{scale}"
            result = try_decode_dmtx(scaled, label)
            if result:
                cv2.imwrite(f"debug/cropped_dmtx_success_{label}.png", scaled)

# Also try on raw + clahe
for name, img in [("raw", raw), ("clahe", clahe), ("sharp", sharpen)]:
    for scale in (2, 3):
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        label = f"dmtx_{name}_x{scale}"
        result = try_decode_dmtx(scaled, label)
        if result:
            cv2.imwrite(f"debug/cropped_dmtx_success_{label}.png", scaled)

print("\nDone!")
