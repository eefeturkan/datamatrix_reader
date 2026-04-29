"""Try pylibdmtx with aggressive settings + various preprocessing on cropped.png"""
import cv2
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from pylibdmtx.pylibdmtx import decode as dmtx_decode
import zxingcpp

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

smooth = cv2.bilateralFilter(raw, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)
sharpen = cv2.addWeighted(clahe, 2.0, cv2.GaussianBlur(clahe, (0, 0), 1.5), -1.0, 0)

# Preprocess variants
variants = {}
variants["raw"] = raw
variants["clahe"] = clahe
variants["sharpen"] = sharpen
variants["smooth"] = smooth
variants["inv_raw"] = 255 - raw
variants["inv_clahe"] = 255 - clahe
variants["inv_sharpen"] = 255 - sharpen

# Threshold variants
for block in (21, 31, 41):
    for c in (3, 5):
        adapt = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, c)
        variants[f"adapt_b{block}_c{c}"] = adapt
        variants[f"adapt_inv_b{block}_c{c}"] = 255 - adapt

# Otsu on tophat
for ks in (13, 17, 21):
    tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants[f"tophat_otsu_k{ks}"] = binary
    variants[f"tophat_otsu_inv_k{ks}"] = 255 - binary

# Morphological background subtraction
for ks in (21, 31):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bg = cv2.morphologyEx(clahe, cv2.MORPH_CLOSE, kernel)
    sub = cv2.subtract(bg, clahe)
    sub_norm = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    variants[f"morphsub_k{ks}"] = sub_norm
    variants[f"morphsub_inv_k{ks}"] = 255 - sub_norm

print(f"Total variants: {len(variants)}")

# Try each variant with pylibdmtx (generous timeout)
print("\n=== pylibdmtx (2s timeout per variant) ===")
for name, img in variants.items():
    for scale in (1, 2, 3):
        scaled = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        t0 = time.time()
        try:
            results = dmtx_decode(scaled, timeout=2000, max_count=3, corrections=10)
            for r in (results or []):
                text = r.data.decode("utf-8", "replace").strip()
                if text:
                    print(f"  DMTX [{name} x{scale}]: '{text}' ({time.time()-t0:.1f}s)")
                    cv2.imwrite(f"debug/cropped_dmtx_success_{name}_x{scale}.png", scaled)
        except Exception as e:
            pass

# Try zxing with try_harder options
print("\n=== zxing comprehensive ===")
for name, img in variants.items():
    for scale in (1, 2, 3):
        scaled = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_CUBIC)
        try:
            results = zxingcpp.read_barcodes(
                scaled,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=True,
                text_mode=zxingcpp.TextMode.Plain,
            )
            for r in (results or []):
                text = (getattr(r, "text", "") or "").strip()
                if text:
                    print(f"  ZXING [{name} x{scale}]: '{text}'")
                    cv2.imwrite(f"debug/cropped_zxing_success_{name}_x{scale}.png", scaled)
        except:
            pass

print("\nDone!")
