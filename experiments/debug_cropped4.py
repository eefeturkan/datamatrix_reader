"""Try blob-centroid based grid fitting for cropped.png"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"d:\DATAGUESS\datamatrix_v2\src")))
from datamatrix_reader.pipeline import (
    _orient_bits, _render_bits, _decode_pure_render
)

raw = cv2.imread(r"d:\DATAGUESS\datamatrix_v2\cropped.png", cv2.IMREAD_GRAYSCALE)
h, w = raw.shape
print(f"Image: {w}x{h}")

def detect_dot_centroids(roi):
    """Detect dot centroids using adaptive morphological approach."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(roi)
    
    centroids_all = []
    
    # Try multiple response maps
    for sigma in (2.0, 3.0):
        blurred = cv2.GaussianBlur(clahe.astype(np.float32), (0, 0), sigma)
        log = -cv2.Laplacian(blurred, cv2.CV_32F)
        log_pos = np.clip(log, 0, None)
        norm = cv2.normalize(log_pos, None, 0, 1, cv2.NORM_MINMAX)
        
        for q in (0.90, 0.92, 0.94):
            thresh_val = np.quantile(norm, q)
            binary = (norm >= thresh_val).astype(np.uint8)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            )
            cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
            
            points = []
            for i in range(1, cc):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if 3 <= area <= 200:
                    cx, cy = cents[i]
                    points.append((float(cx), float(cy)))
            
            centroids_all.append((f"log_{sigma}_q{q:.2f}", points))
    
    # Also try tophat based
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    tophat_norm = cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)
    
    for q in (0.90, 0.92, 0.94):
        thresh_val = np.quantile(tophat_norm, q)
        binary = (tophat_norm >= thresh_val).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )
        cc, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
        
        points = []
        for i in range(1, cc):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 3 <= area <= 200:
                cx, cy = cents[i]
                points.append((float(cx), float(cy)))
        
        centroids_all.append((f"tophat_q{q:.2f}", points))
    
    return centroids_all

def cluster_grid(points, module_count=20):
    """Try to find a regular grid from a set of points."""
    if len(points) < module_count * module_count * 0.3:
        return None
    
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Estimate pitch from histogram of pairwise distances
    # Only use nearby points to avoid combinatorial explosion
    from scipy.spatial import KDTree
    tree = KDTree(np.column_stack([xs, ys]))
    
    # For each point, find nearest neighbors
    nn_dists = []
    for i in range(len(points)):
        dists, _ = tree.query(points[i], k=min(6, len(points)))
        nn_dists.extend(dists[1:])  # exclude self
    
    nn_dists = np.array(nn_dists)
    nn_dists = nn_dists[(nn_dists > 5) & (nn_dists < 50)]
    
    if len(nn_dists) < 20:
        return None
    
    # Histogram to find dominant pitch
    hist, bins = np.histogram(nn_dists, bins=100)
    peak_idx = np.argmax(hist)
    pitch_estimate = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    return pitch_estimate

def try_grid_reconstruction(roi, points, pitch, module_count=20):
    """Given dot centroids and pitch estimate, try to fit a 20x20 grid and reconstruct."""
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Cluster y-coordinates into rows
    ys_sorted = np.sort(ys)
    rows = []
    current_row = [ys_sorted[0]]
    for y in ys_sorted[1:]:
        if y - current_row[-1] < pitch * 0.4:
            current_row.append(y)
        else:
            rows.append(np.mean(current_row))
            current_row = [y]
    rows.append(np.mean(current_row))
    
    print(f"  Found {len(rows)} y-rows")
    
    if len(rows) < module_count - 4:
        return None
    
    # For each row, collect x-coordinates
    row_points = []
    for row_y in rows:
        mask = np.abs(ys - row_y) < pitch * 0.4
        row_xs = sorted(xs[mask])
        row_points.append(row_xs)
    
    # Find the rows with most points (likely the solid borders)
    row_counts = [(i, len(rp)) for i, rp in enumerate(row_points)]
    row_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top row counts: {row_counts[:5]}")
    
    # Find a run of ~20 consecutive rows with pitch spacing
    # First, find best starting row
    best_grid = None
    best_score = -1
    
    for start_idx in range(len(rows)):
        if start_idx + module_count - 1 >= len(rows):
            # Check if remaining rows can be extended
            pass
        
        # Try to pick module_count rows starting from start_idx with ~pitch spacing
        grid_rows = [rows[start_idx]]
        for ri in range(start_idx + 1, len(rows)):
            expected_y = grid_rows[-1] + pitch
            if abs(rows[ri] - expected_y) < pitch * 0.35:
                grid_rows.append(rows[ri])
            elif rows[ri] > expected_y + pitch * 0.5:
                break
            if len(grid_rows) >= module_count:
                break
        
        if len(grid_rows) < module_count - 3:
            continue
        
        # For grid columns, find the most common x-positions
        all_xs_in_grid = []
        for row_y in grid_rows:
            mask = np.abs(ys - row_y) < pitch * 0.4
            all_xs_in_grid.extend(xs[mask].tolist())
        
        if not all_xs_in_grid:
            continue
        
        # Cluster x-coordinates into columns
        all_xs_sorted = sorted(all_xs_in_grid)
        cols = []
        current_col = [all_xs_sorted[0]]
        for x in all_xs_sorted[1:]:
            if x - current_col[-1] < pitch * 0.4:
                current_col.append(x)
            else:
                cols.append(np.mean(current_col))
                current_col = [x]
        cols.append(np.mean(current_col))
        
        if len(cols) < module_count - 3:
            continue
        
        # Score this grid candidate
        score = len(grid_rows) + len(cols) * 0.5
        if score > best_score:
            best_score = score
            best_grid = (grid_rows, cols, pitch)
    
    if best_grid is None:
        return None
    
    grid_rows, grid_cols, grid_pitch = best_grid
    print(f"  Best grid: {len(grid_rows)} rows x {len(grid_cols)} cols, pitch={grid_pitch:.1f}")
    
    # Now build the bit matrix by checking if there's a dot at each grid position
    # Use a response map for scoring
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(roi)
    blurred = cv2.GaussianBlur(clahe.astype(np.float32), (0, 0), 2.5)
    log = -cv2.Laplacian(blurred, cv2.CV_32F)
    log_pos = np.clip(log, 0, None)
    response = cv2.normalize(log_pos, None, 0, 1, cv2.NORM_MINMAX)
    
    # Also try tophat
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    tophat_resp = cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)
    
    results = []
    
    for resp_name, resp in [("log", response), ("tophat", tophat_resp)]:
        # Pad grid rows/cols to exactly module_count if needed
        if len(grid_rows) < module_count:
            # Extrapolate
            while len(grid_rows) < module_count:
                grid_rows.append(grid_rows[-1] + grid_pitch)
        if len(grid_cols) < module_count:
            while len(grid_cols) < module_count:
                grid_cols.append(grid_cols[-1] + grid_pitch)
        
        grid_rows = grid_rows[:module_count]
        grid_cols = grid_cols[:module_count]
        
        # Sample response at grid points
        radius = max(3, int(grid_pitch * 0.2))
        scores = np.zeros((module_count, module_count), dtype=np.float32)
        for ri, ry in enumerate(grid_rows):
            for ci, cx in enumerate(grid_cols):
                iy = int(round(ry))
                ix = int(round(cx))
                y0 = max(0, iy - radius)
                y1 = min(roi.shape[0], iy + radius + 1)
                x0 = max(0, ix - radius)
                x1 = min(roi.shape[1], ix + radius + 1)
                patch = resp[y0:y1, x0:x1]
                scores[ri, ci] = float(patch.max()) if patch.size else 0.0
        
        for sq in (0.40, 0.45, 0.50, 0.55):
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
                print(f"  DECODED with {resp_name} sq={sq:.2f}! score={score:.2f}: {decoded}")
                cv2.imwrite(f"debug/cropped_grid_{resp_name}_{sq:.2f}.png", rendered)
                results.append((decoded, score, rendered, oriented))
            elif score > 3.5:
                # Save for debugging even if not decoded
                cv2.imwrite(f"debug/cropped_grid_nodecode_{resp_name}_{sq:.2f}.png", rendered)
    
    return results

# Detect centroids
all_centroids = detect_dot_centroids(raw)

for name, points in all_centroids:
    print(f"\n--- {name}: {len(points)} points ---")
    
    if len(points) < 50:
        continue
    
    pitch = cluster_grid(points, 20)
    if pitch is None:
        print("  No pitch found")
        continue
    
    print(f"  Estimated pitch: {pitch:.1f}")
    
    results = try_grid_reconstruction(raw, points, pitch)
    if results:
        for decoded, score, rendered, oriented in results:
            print(f"  -> {decoded} (score={score:.2f})")

# Also save a visualization of detected dots
for name, points in all_centroids[:3]:
    vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), 1)
    cv2.imwrite(f"debug/cropped_dots_{name}.png", vis)

print("\nDone!")
