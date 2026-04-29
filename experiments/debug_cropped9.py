"""Key insight: downscale cropped.png to match test4/test5 dot density, then use 
the EXISTING reconstruction pipeline directly. The existing pipeline works for 
images where dot pitch is ~10-16px. We just need to resize cropped.png to bring
its pitch into that range."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader import decode_image

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png")
h, w = raw.shape[:2]
print(f"Original: {w}x{h}")

# test4.png is ~312x316 and has pitch ~13-14px
# cropped.png is 464x423
# So we need to resize to roughly 280-320 range

for target_w in range(220, 360, 10):
    scale = target_w / w
    resized = cv2.resize(raw, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Encode as bytes
    _, buf = cv2.imencode(".png", resized)
    image_bytes = buf.tobytes()
    
    result = decode_image(image_bytes)
    
    if result.text:
        print(f"  target_w={target_w}: DECODED '{result.text}' via {result.engine} ({result.stage_name})")
        cv2.imwrite(f"debug/cropped_resize_{target_w}_decoded.png", result.processed_image)
    else:
        print(f"  target_w={target_w}: no decode")

# Also try with preprocessing before resize
print("\n=== With preprocessing ===")
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
smoothed = cv2.bilateralFilter(gray, 11, 75, 75)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smoothed)

for target_w in range(240, 340, 10):
    scale = target_w / w
    
    for name, src in [("bilateral", smoothed), ("clahe", clahe)]:
        resized = cv2.resize(src, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Convert back to BGR for encode (the pipeline expects color or grayscale)
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", resized_bgr)
        image_bytes = buf.tobytes()
        
        result = decode_image(image_bytes)
        
        if result.text:
            print(f"  {name} target_w={target_w}: DECODED '{result.text}' via {result.engine}")
            cv2.imwrite(f"debug/cropped_{name}_{target_w}_decoded.png", result.processed_image)

# Try inverted
print("\n=== Inverted ===")
for target_w in range(240, 340, 20):
    scale = target_w / w
    inverted = 255 - gray
    resized = cv2.resize(inverted, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".png", resized_bgr)
    result = decode_image(buf.tobytes())
    if result.text:
        print(f"  inverted target_w={target_w}: DECODED '{result.text}'")

print("\nDone!")
