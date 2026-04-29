from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step07_combined_edge_fit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Fit:
    score: float
    origin: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    left: int
    top: int
    edge_score: float
    inside_score: float
    outside_score: float


def extract_red_centers(marked_bgr: np.ndarray) -> np.ndarray:
    b = marked_bgr[:, :, 0].astype(np.int16)
    g = marked_bgr[:, :, 1].astype(np.int16)
    r = marked_bgr[:, :, 2].astype(np.int16)
    mask = (r > 160) & (r - g > 60) & (r - b > 60)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    pts: list[tuple[float, float]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 400:
            x, y = centroids[idx]
            pts.append((float(x), float(y)))
    return np.array(pts, dtype=np.float32)


def local_contrast_normalize(gray: np.ndarray, win: int = 41, eps: float = 1e-3) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(gray_f, (0, 0), win / 6.0)
    sq_mu = cv2.GaussianBlur(gray_f * gray_f, (0, 0), win / 6.0)
    sigma = np.sqrt(np.maximum(sq_mu - mu * mu, 0.0))
    return (gray_f - mu) / (sigma + eps)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def hough_centers(gray_inv: np.ndarray) -> np.ndarray:
    blur = cv2.medianBlur(gray_inv, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        1.0,
        10,
        param1=80,
        param2=10,
        minRadius=2,
        maxRadius=8,
    )
    if circles is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.round(circles[0, :, :2]).astype(np.float32)


def hessian_blob_multi(dark_lcn: np.ndarray) -> np.ndarray:
    maps = []
    for sigma in (1.4, 1.8, 2.2, 2.8):
        smooth = cv2.GaussianBlur(dark_lcn, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        det = dxx * dyy - dxy * dxy
        trace = dxx + dyy
        disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
        l1 = 0.5 * (trace + disc)
        l2 = 0.5 * (trace - disc)
        same_sign = (l1 > 0) & (l2 > 0)
        isotropy = np.minimum(np.abs(l1), np.abs(l2)) / (np.maximum(np.abs(l1), np.abs(l2)) + 1e-6)
        blobness = np.where(same_sign, np.sqrt(np.maximum(det, 0.0)) * isotropy, 0.0)
        maps.append(normalize_u8(blobness))
    return np.max(np.stack(maps, axis=0), axis=0).astype(np.uint8)


def fft_best_map(dark_lcn: np.ndarray) -> np.ndarray:
    h, w = dark_lcn.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX * FX + FY * FY)
    f0 = 1.0 / 17.0
    width = 0.014
    mask = np.exp(-0.5 * ((R - f0) / width) ** 2)
    spectrum = np.fft.fft2(dark_lcn)
    recon = np.fft.ifft2(spectrum * mask)
    return normalize_u8(np.abs(recon))


def topk_points(map_u8: np.ndarray, k: int, min_dist: float = 8.0) -> np.ndarray:
    ys, xs = np.where(map_u8 > 0)
    scores = map_u8[ys, xs].astype(np.float32)
    order = np.argsort(scores)[::-1]
    pts: list[tuple[float, float]] = []
    r2 = min_dist * min_dist
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        if all((x - px) ** 2 + (y - py) ** 2 > r2 for px, py in pts):
            pts.append((x, y))
            if len(pts) >= k:
                break
    return np.array(pts, dtype=np.float32)


def build_weighted_points(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    lcn = local_contrast_normalize(gray, win=41)
    dark = np.clip(-lcn, 0.0, None)
    dark_u8 = normalize_u8(dark)
    gray_inv = 255 - gray
    hough = hough_centers(gray_inv)
    hessian = topk_points(hessian_blob_multi(dark), 220)
    fft = topk_points(fft_best_map(dark), 220)
    points: list[np.ndarray] = []
    weights: list[float] = []
    for p in hough:
        points.append(p)
        weights.append(1.0)
    for p in hessian:
        points.append(p)
        weights.append(0.85)
    for p in fft:
        points.append(p)
        weights.append(0.80)
    merged_points, merged_weights = merge_weighted_points(np.array(points, dtype=np.float32), np.array(weights, dtype=np.float32), radius=5.5)
    return merged_points, merged_weights, {
        "gray_inv": gray_inv,
        "dark_u8": dark_u8,
    }


def merge_weighted_points(points: np.ndarray, weights: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(weights)[::-1]
    kept_pts: list[np.ndarray] = []
    kept_w: list[float] = []
    r2 = radius * radius
    for idx in order:
        p = points[idx]
        w = float(weights[idx])
        merged = False
        for j, kp in enumerate(kept_pts):
            if float(np.sum((p - kp) ** 2)) <= r2:
                total = kept_w[j] + w
                kept_pts[j] = (kp * kept_w[j] + p * w) / total
                kept_w[j] = total
                merged = True
                break
        if not merged:
            kept_pts.append(p.copy())
            kept_w.append(w)
    return np.array(kept_pts, dtype=np.float32), np.array(kept_w, dtype=np.float32)


def candidate_bases() -> list[tuple[np.ndarray, np.ndarray]]:
    bases: list[tuple[np.ndarray, np.ndarray]] = []
    for ax in (3.0, 3.8, 4.6):
        for ay in (95.0, 96.5, 98.0):
            for sx in (15.8, 16.2):
                for sy in (16.0, 16.6):
                    vx = np.array([math.cos(math.radians(ax)) * sx, math.sin(math.radians(ax)) * sx], dtype=np.float32)
                    vy = np.array([math.cos(math.radians(ay)) * sy, math.sin(math.radians(ay)) * sy], dtype=np.float32)
                    bases.append((vx, vy))
    return bases


def fit_edges(points: np.ndarray, weights: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> Fit:
    basis = np.column_stack((vx, vy))
    inv = np.linalg.inv(basis)
    coords = points @ inv.T
    best: Fit | None = None

    for off_u in np.linspace(0.0, 0.92, 12):
        for off_v in np.linspace(0.0, 0.92, 12):
            shifted = coords - np.array([off_u, off_v], dtype=np.float32)
            rounded = np.rint(shifted)
            err = np.linalg.norm(shifted - rounded, axis=1)
            good = err <= 0.34
            if int(np.sum(good)) < 120:
                continue
            ij = rounded[good].astype(np.int32)
            w = weights[good]
            imin, imax = int(ij[:, 0].min()), int(ij[:, 0].max())
            jmin, jmax = int(ij[:, 1].min()), int(ij[:, 1].max())
            i_candidates = list(range(imin, imax - 18))
            j_candidates = list(range(jmin, jmax - 18))
            if not i_candidates or not j_candidates:
                continue
            i_scores = []
            for left in i_candidates:
                right = left + 19
                left_score = float(w[ij[:, 0] == left].sum())
                right_score = float(w[ij[:, 0] == right].sum())
                i_scores.append((left_score + right_score, left))
            j_scores = []
            for top in j_candidates:
                bottom = top + 19
                top_score = float(w[ij[:, 1] == top].sum())
                bottom_score = float(w[ij[:, 1] == bottom].sum())
                j_scores.append((top_score + bottom_score, top))

            top_lefts = [left for _, left in sorted(i_scores, reverse=True)[:5]]
            top_tops = [top for _, top in sorted(j_scores, reverse=True)[:5]]

            for left in top_lefts:
                right = left + 19
                i_ok = (ij[:, 0] >= left) & (ij[:, 0] <= right)
                if np.sum(i_ok) < 70:
                    continue
                for top in top_tops:
                    bottom = top + 19
                    in_box = i_ok & (ij[:, 1] >= top) & (ij[:, 1] <= bottom)
                    if np.sum(in_box) < 90:
                        continue
                    box_ij = ij[in_box]
                    box_w = w[in_box]
                    left_score = float(box_w[box_ij[:, 0] == left].sum())
                    right_score = float(box_w[box_ij[:, 0] == right].sum())
                    top_score = float(box_w[box_ij[:, 1] == top].sum())
                    bottom_score = float(box_w[box_ij[:, 1] == bottom].sum())
                    edge_score = left_score + right_score + top_score + bottom_score
                    inside_score = float(box_w.sum())
                    outside_score = float(w[~in_box].sum())

                    score = 8.0 * edge_score + 0.8 * inside_score - 0.85 * outside_score
                    origin_idx = np.array([left + off_u, top + off_v], dtype=np.float32)
                    origin = origin_idx @ basis.T
                    fit = Fit(score, origin.astype(np.float32), vx, vy, left, top, edge_score, inside_score, outside_score)
                    if best is None or fit.score > best.score:
                        best = fit
    if best is None:
        raise RuntimeError("No edge fit found.")
    return best


def draw_overlay(base: np.ndarray, truth: np.ndarray, fit: Fit, path: Path) -> None:
    vis = base.copy()
    for x, y in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for row in range(20):
        for col in range(20):
            pt = fit.origin + col * fit.vx + row * fit.vy
            color = (0, 255, 0)
            if row in (0, 19) or col in (0, 19):
                color = (0, 255, 255)
            cv2.circle(vis, (int(round(float(pt[0]))), int(round(float(pt[1])))), 3, color, 1, lineType=cv2.LINE_AA)
    tl = fit.origin
    tr = fit.origin + 19 * fit.vx
    bl = fit.origin + 19 * fit.vy
    br = fit.origin + 19 * fit.vx + 19 * fit.vy
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [quad], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def edge_recall(truth: np.ndarray, fit: Fit) -> dict[str, float]:
    basis = np.column_stack((fit.vx, fit.vy))
    inv = np.linalg.inv(basis)
    local = (truth - fit.origin[None, :]) @ inv.T
    rounded = np.rint(local)
    err = np.linalg.norm(local - rounded, axis=1)
    good = err <= 0.36
    ij = rounded.astype(np.int32)
    edge_mask = good & (
        (ij[:, 0] == 0)
        | (ij[:, 0] == 19)
        | (ij[:, 1] == 0)
        | (ij[:, 1] == 19)
    )
    return {
        "truth_good": float(np.sum(good)),
        "truth_edge_good": float(np.sum(edge_mask)),
        "truth_mean_err": float(err[good].mean()) if np.any(good) else 999.0,
        "truth_min_i": float(ij[good, 0].min()) if np.any(good) else 999.0,
        "truth_max_i": float(ij[good, 0].max()) if np.any(good) else -999.0,
        "truth_min_j": float(ij[good, 1].min()) if np.any(good) else 999.0,
        "truth_max_j": float(ij[good, 1].max()) if np.any(good) else -999.0,
    }


def main() -> None:
    raw = cv2.imread(str(ROOT / "cropped.png"))
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    if raw is None or marked is None:
        raise SystemExit("Missing input images.")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    truth = extract_red_centers(marked)

    points, weights, images = build_weighted_points(gray)
    cv2.imwrite(str(OUT_DIR / "gray_inv.png"), images["gray_inv"])
    cv2.imwrite(str(OUT_DIR / "dark_u8.png"), images["dark_u8"])

    best: Fit | None = None
    for vx, vy in candidate_bases():
        try:
            fit = fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit

    if best is None:
        raise SystemExit("No combined edge fit found.")

    draw_overlay(marked, truth, best, OUT_DIR / "combined_edge_overlay.png")
    stats = edge_recall(truth, best)
    lines = [
        f"point_count={len(points)}",
        f"origin=({best.origin[0]:.3f},{best.origin[1]:.3f})",
        f"vx=({best.vx[0]:.3f},{best.vx[1]:.3f}) len={np.linalg.norm(best.vx):.3f} angle={math.degrees(math.atan2(float(best.vx[1]), float(best.vx[0]))):.3f}",
        f"vy=({best.vy[0]:.3f},{best.vy[1]:.3f}) len={np.linalg.norm(best.vy):.3f} angle={math.degrees(math.atan2(float(best.vy[1]), float(best.vy[0]))):.3f}",
        f"left={best.left}",
        f"top={best.top}",
        f"score={best.score:.3f}",
        f"edge_score={best.edge_score:.3f}",
        f"inside_score={best.inside_score:.3f}",
        f"outside_score={best.outside_score:.3f}",
    ]
    for k, v in stats.items():
        lines.append(f"{k}={v:.3f}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
