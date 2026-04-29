"""Use FFT/frequency analysis to find true grid period, then vote-based grid alignment."""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import _orient_bits, _render_bits, _decode_pure_render

resp_img = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\debug\cropped_morphsub_31.png", cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = resp_img.shape

# ---------------------------------------------------------------
# Approach: FFT on the response map to find the true dot period
# ---------------------------------------------------------------
resp_f = resp_img.astype(np.float32) / 255.0

# FFT on columns (horizontal period)
fft_cols = np.fft.rfft(resp_f.mean(axis=0))
freqs_h = np.fft.rfftfreq(w)
# Find dominant frequency (skip DC)
mag_h = np.abs(fft_cols)
mag_h[0] = 0  # remove DC
# Smooth
mag_h = cv2.GaussianBlur(mag_h.reshape(1, -1), (7, 1), 0).flatten()
peak_idx_h = int(np.argmax(mag_h[1:]) + 1)
period_h = 1.0 / freqs_h[peak_idx_h] if freqs_h[peak_idx_h] > 0 else w
print(f"FFT horizontal period: {period_h:.1f}px (freq={freqs_h[peak_idx_h]:.4f})")

fft_rows = np.fft.rfft(resp_f.mean(axis=1))
freqs_v = np.fft.rfftfreq(h)
mag_v = np.abs(fft_rows)
mag_v[0] = 0
mag_v = cv2.GaussianBlur(mag_v.reshape(1, -1), (7, 1), 0).flatten()
peak_idx_v = int(np.argmax(mag_v[1:]) + 1)
period_v = 1.0 / freqs_v[peak_idx_v] if freqs_v[peak_idx_v] > 0 else h
print(f"FFT vertical period: {period_v:.1f}px (freq={freqs_v[peak_idx_v]:.4f})")

# Show several candidate periods
print("\nTop horizontal candidates:")
top_h = np.argsort(mag_h[1:])[-10:] + 1
for idx in reversed(top_h):
    if freqs_h[idx] > 0:
        print(f"  period={1/freqs_h[idx]:.1f}px  mag={mag_h[idx]:.4f}")

# ---------------------------------------------------------------
# Use the FFT period as pitch and do Radon-style phase search
# ---------------------------------------------------------------
pitch = (period_h + period_v) / 2
print(f"\nUsing pitch: {pitch:.1f}px")

# Phase search: for each x-offset in [0, pitch), count how many pixels 
# in the response map align with a regular grid column
def score_phase(resp, pitch, axis=1):
    """Score each phase offset for how well it aligns with grid."""
    scores = []
    n = resp.shape[axis]
    proj = resp.mean(axis=1-axis)  # project onto the axis
    n_phases = int(round(pitch))
    for phase in np.linspace(0, pitch, n_phases, endpoint=False):
        positions = np.arange(phase, n, pitch)
        positions = positions[positions < n].astype(int)
        score = proj[positions].sum()
        scores.append((phase, score))
    return max(scores, key=lambda x: x[1])

best_x_phase, score_x = score_phase(resp_f, pitch, axis=1)
best_y_phase, score_y = score_phase(resp_f, pitch, axis=0)
print(f"Best x-phase: {best_x_phase:.1f} (score={score_x:.3f})")
print(f"Best y-phase: {best_y_phase:.1f} (score={score_y:.3f})")

# Try multiple pitch candidates around the FFT estimate
best_decoded = None
for pitch_try in np.arange(period_h * 0.85, period_h * 1.15, 0.5):
    for x_off in np.arange(0, pitch_try, pitch_try / 8):
        for y_off in np.arange(0, pitch_try, pitch_try / 8):
            # Build grid bit matrix
            bits = np.zeros((20, 20), dtype=np.uint8)
            for row in range(20):
                gy = y_off + row * pitch_try
                if gy >= h: break
                for col in range(20):
                    gx = x_off + col * pitch_try
                    if gx >= w: break
                    # Sample response in a window
                    pr = max(3, int(pitch_try * 0.25))
                    y0 = max(0, int(gy) - pr); y1 = min(h, int(gy) + pr + 1)
                    x0 = max(0, int(gx) - pr); x1 = min(w, int(gx) + pr + 1)
                    patch = resp_f[y0:y1, x0:x1]
                    bits[row, col] = 1 if (patch.max() > 0.15) else 0
            
            for r in range(4):
                tb = np.rot90(bits, r).copy()
                tb[0, :] = 1; tb[:, 0] = 1
                rendered = _render_bits(tb)
                decoded = _decode_pure_render(rendered)
                if decoded:
                    print(f"\nDECODED! pitch={pitch_try:.1f} x0={x_off:.0f} y0={y_off:.0f} rot={r*90}: {decoded}")
                    cv2.imwrite("debug/cropped_fft_DECODED.png", rendered)
                    best_decoded = decoded
                    break
            if best_decoded: break
        if best_decoded: break
    if best_decoded: break

# ---------------------------------------------------------------
# Also: try row-by-row intensity profile to find exact y-offsets
# ---------------------------------------------------------------
if not best_decoded:
    print("\n\nRow-by-row approach...")
    # For each candidate pitch and y-offset, sum response at expected row positions
    row_proj = resp_f.mean(axis=1)
    col_proj = resp_f.mean(axis=0)
    
    print(f"Row projection max at: {np.argmax(row_proj)}")
    print(f"Col projection max at: {np.argmax(col_proj)}")
    
    # Find top-20 peaks in row projection
    from datamatrix_reader.pipeline import _find_peaks_1d
    row_peaks = _find_peaks_1d(row_proj, min_dist=int(period_v * 0.5), 
                                threshold=float(np.quantile(row_proj, 0.5)))
    col_peaks = _find_peaks_1d(col_proj, min_dist=int(period_h * 0.5),
                                threshold=float(np.quantile(col_proj, 0.5)))
    print(f"Row peaks ({len(row_peaks)}): {row_peaks}")
    print(f"Col peaks ({len(col_peaks)}): {col_peaks}")
    
    if len(row_peaks) >= 5 and len(col_peaks) >= 5:
        y_pitch = float(np.median(np.diff(row_peaks)))
        x_pitch = float(np.median(np.diff(col_peaks)))
        print(f"Row pitch: {y_pitch:.1f}, Col pitch: {x_pitch:.1f}")
        
        # Build bit matrix using these peaks as grid lines
        # Extend to 20x20
        while len(row_peaks) < 20:
            row_peaks.append(int(row_peaks[-1] + y_pitch))
        while len(col_peaks) < 20:
            col_peaks.append(int(col_peaks[-1] + x_pitch))
        row_peaks = row_peaks[:20]
        col_peaks = col_peaks[:20]
        
        pr = max(3, int(min(x_pitch, y_pitch) * 0.25))
        bits = np.zeros((20, 20), dtype=np.uint8)
        for ri, ry in enumerate(row_peaks):
            for ci, cx in enumerate(col_peaks):
                y0 = max(0, ry - pr); y1 = min(h, ry + pr + 1)
                x0 = max(0, cx - pr); x1 = min(w, cx + pr + 1)
                patch = resp_f[y0:y1, x0:x1]
                bits[ri, ci] = 1 if (patch.max() > 0.15) else 0
        
        for r in range(4):
            tb = np.rot90(bits, r).copy()
            tb[0, :] = 1; tb[:, 0] = 1
            rendered = _render_bits(tb)
            cv2.imwrite(f"debug/cropped_rowcol_r{r*90}.png", rendered)
            decoded = _decode_pure_render(rendered)
            if decoded:
                print(f"DECODED with row/col peaks r={r*90}!: {decoded}")
                cv2.imwrite("debug/cropped_rowcol_DECODED.png", rendered)

print("\nDone!")
