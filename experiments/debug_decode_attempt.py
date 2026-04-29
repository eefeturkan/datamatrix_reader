"""Try to decode the bestmatrix renders with all available decoders."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _decode_pure_render
from pylibdmtx.pylibdmtx import decode as dmtx_decode
import zxingcpp

for rot in (0, 90, 180, 270):
    img = cv2.imread(f"debug/cropped_bestmatrix_r{rot}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    
    print(f"\n=== r{rot} ===")
    
    # Try pipeline decoder
    dec = _decode_pure_render(img)
    if dec:
        print(f"  pipeline: {dec}")
    
    # Try zxing
    for scale in (1, 2, 3):
        scaled = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for inv in (False, True):
            test = 255 - scaled if inv else scaled
            try:
                results = zxingcpp.read_barcodes(test,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True, try_downscale=True,
                    text_mode=zxingcpp.TextMode.Plain)
                for r in (results or []):
                    text = (getattr(r, "text", "") or "").strip()
                    if text:
                        print(f"  zxing x{scale}{'_inv' if inv else ''}: {text}")
            except:
                pass
    
    # Try dmtx
    for scale in (2, 3, 4):
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for inv in (False, True):
            test = 255 - scaled if inv else scaled
            try:
                results = dmtx_decode(test, timeout=2000, max_count=3, corrections=10)
                for r in (results or []):
                    text = r.data.decode("utf-8", "replace").strip()
                    if text:
                        print(f"  dmtx x{scale}{'_inv' if inv else ''}: {text}")
            except:
                pass

# Also try the renders with borders on all 4 sides
print("\n=== With full borders ===")
img = cv2.imread("debug/cropped_bestmatrix_r0.png", cv2.IMREAD_GRAYSCALE)
if img is not None:
    # Convert to bits
    from datamatrix_reader.pipeline import _orient_bits, _render_bits
    h, w = img.shape
    bits = (img < 128).astype(np.uint8)
    print(f"Bits shape: {bits.shape}")
    print(f"Row 0 (should be solid): {bits[0, :].sum()}/20")
    print(f"Col 0 (should be solid): {bits[:, 0].sum()}/20")
    print(f"Row 19 (should be alternating): {bits[19, :]}")
    print(f"Col 19 (should be alternating): {bits[:, 19]}")

print("\nDone!")
