from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step04_hybrid_frame_fit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FitResult:
    score: float
    offset_u: float
    offset_v: float
    origin: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    good_count: int
    mean_err: float
    span_u: int
    span_v: int


def extract_red_centers(marked_bgr: np.ndarray) -> np.ndarray:
    b = marked_bgr[:, :, 0].astype(np.int16)
    g = marked_bgr[:, :, 1].astype(np.int16)
    r = marked_bgr[:, :, 2].astype(np.int16)
    mask = (r > 160) & (r - g > 60) & (r - b > 60)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    points: list[tuple[float, float]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 400:
            x, y = centroids[idx]
            points.append((float(x), float(y)))
    return np.array(points, dtype=np.float32)


def build_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32) / 255.0
    median31 = cv2.medianBlur(gray, 31).astype(np.float32) / 255.0
    median_dark_31 = np.clip(median31 - gray_f, 0.0, 1.0)
    radius = 5
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    sigma = 2.1
    ring = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ring /= float(ring.sum())
    matched_r5_pos = cv2.filter2D(median_dark_31, cv2.CV_32F, ring)
    matched_r5_pos = cv2.GaussianBlur(matched_r5_pos, (0, 0), 0.9)
    gray_inv = 255 - gray
    matched_u8 = cv2.normalize(matched_r5_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray_inv, matched_u8


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


def matched_peaks(image: np.ndarray) -> np.ndarray:
    f = image.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(f, (0, 0), 1.0)
    dil = cv2.dilate(blur, np.ones((5, 5), np.uint8))
    mask = (blur >= dil - 1e-6) & (blur >= np.quantile(blur, 0.955))
    ys, xs = np.where(mask)
    scores = blur[ys, xs]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float]] = []
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        if all((x - px) ** 2 + (y - py) ** 2 > 8.5 ** 2 for px, py in selected):
            selected.append((x, y))
    return np.array(selected, dtype=np.float32)


def merge_points(a: np.ndarray, b: np.ndarray, radius: float = 7.0) -> np.ndarray:
    merged: list[np.ndarray] = [pt.copy() for pt in a]
    for pt in b:
        if len(merged) == 0:
            merged.append(pt.copy())
            continue
        arr = np.array(merged, dtype=np.float32)
        d = np.linalg.norm(arr - pt[None, :], axis=1)
        j = int(np.argmin(d))
        if d[j] <= radius:
            merged[j] = ((arr[j] + pt) * 0.5).astype(np.float32)
        else:
            merged.append(pt.copy())
    return np.array(merged, dtype=np.float32)


def estimate_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = points[None, :, :] - points[:, None, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argpartition(dist, kth=8, axis=1)[:, :8]

    vectors: list[np.ndarray] = []
    for i, nbrs in enumerate(nearest):
        p = points[i]
        for j in nbrs:
            v = points[j] - p
            d = float(np.linalg.norm(v))
            if 12.0 <= d <= 22.0:
                if v[0] < 0:
                    v = -v
                vectors.append(v)

    arr = np.array(vectors, dtype=np.float32)
    angles = np.degrees(np.arctan2(arr[:, 1], arr[:, 0]))
    angles = np.where(angles < 0, angles + 180.0, angles)
    target_x = arr[(angles < 35.0) | (angles > 145.0)]
    target_y = arr[(angles > 55.0) & (angles < 125.0)]
    vx = np.median(target_x, axis=0)
    vy = np.median(target_y, axis=0)
    if vx[0] < 0:
        vx = -vx
    if vy[1] < 0:
        vy = -vy
    return vx.astype(np.float32), vy.astype(np.float32)


def fit_offsets(points: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> FitResult:
    basis = np.column_stack((vx, vy))
    inv = np.linalg.inv(basis)
    coords = points @ inv.T
    best: FitResult | None = None

    for off_u in np.linspace(0.0, 0.95, 20):
        for off_v in np.linspace(0.0, 0.95, 20):
            shifted = coords - np.array([off_u, off_v], dtype=np.float32)
            rounded = np.rint(shifted)
            err = np.linalg.norm(shifted - rounded, axis=1)
            good = err <= 0.34
            if int(good.sum()) < 120:
                continue
            ij = rounded[good].astype(np.int32)
            min_i, max_i = int(ij[:, 0].min()), int(ij[:, 0].max())
            min_j, max_j = int(ij[:, 1].min()), int(ij[:, 1].max())
            span_u = max_i - min_i
            span_v = max_j - min_j
            spread_penalty = abs(span_u - 19) + abs(span_v - 19)
            right_hits = int(np.sum(ij[:, 0] == max_i))
            bottom_hits = int(np.sum(ij[:, 1] == max_j))
            left_hits = int(np.sum(ij[:, 0] == min_i))
            top_hits = int(np.sum(ij[:, 1] == min_j))
            score = (
                float(good.sum())
                - 32.0 * spread_penalty
                + 2.5 * min(right_hits, 8)
                + 2.5 * min(bottom_hits, 8)
                + 2.0 * min(left_hits, 8)
                + 2.0 * min(top_hits, 8)
                - 150.0 * float(err[good].mean())
            )
            origin_index = np.array([min_i, min_j], dtype=np.float32)
            origin = (origin_index + np.array([off_u, off_v], dtype=np.float32)) @ basis.T
            fit = FitResult(
                score=score,
                offset_u=off_u,
                offset_v=off_v,
                origin=origin.astype(np.float32),
                vx=vx,
                vy=vy,
                good_count=int(good.sum()),
                mean_err=float(err[good].mean()),
                span_u=span_u,
                span_v=span_v,
            )
            if best is None or fit.score > best.score:
                best = fit

    if best is None:
        raise RuntimeError("No fit found.")
    return best


def draw_points(base: np.ndarray, red: np.ndarray, green: np.ndarray, path: Path) -> None:
    vis = base.copy()
    for x, y in red:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for x, y in green:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def draw_grid(base: np.ndarray, truth: np.ndarray, fit: FitResult, path: Path) -> None:
    vis = base.copy()
    for x, y in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for row in range(20):
        for col in range(20):
            pt = fit.origin + col * fit.vx + row * fit.vy
            cv2.circle(vis, (int(round(float(pt[0]))), int(round(float(pt[1])))), 3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    tl = fit.origin
    tr = fit.origin + 19 * fit.vx
    bl = fit.origin + 19 * fit.vy
    br = fit.origin + 19 * fit.vx + 19 * fit.vy
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [quad], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def evaluate_truth(truth: np.ndarray, fit: FitResult) -> dict[str, float]:
    basis = np.column_stack((fit.vx, fit.vy))
    inv = np.linalg.inv(basis)
    local = (truth - fit.origin[None, :]) @ inv.T
    rounded = np.rint(local)
    err = np.linalg.norm(local - rounded, axis=1)
    good = err <= 0.36
    ij = rounded[good].astype(np.int32)
    stats = {
        "truth_good": float(np.sum(good)),
        "truth_mean_err": float(err[good].mean()) if np.any(good) else 999.0,
    }
    if np.any(good):
        stats.update(
            {
                "min_i": float(ij[:, 0].min()),
                "max_i": float(ij[:, 0].max()),
                "min_j": float(ij[:, 1].min()),
                "max_j": float(ij[:, 1].max()),
                "right_hits": float(np.sum(ij[:, 0] == ij[:, 0].max())),
                "bottom_hits": float(np.sum(ij[:, 1] == ij[:, 1].max())),
            }
        )
    return stats


def main() -> None:
    raw = cv2.imread(str(ROOT / "cropped.png"))
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    if raw is None or marked is None:
        raise SystemExit("Missing input images.")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    truth = extract_red_centers(marked)
    gray_inv, matched_u8 = build_maps(gray)

    hough = hough_centers(gray_inv)
    peaks = matched_peaks(matched_u8)
    hybrid = merge_points(hough, peaks, radius=7.0)
    vx, vy = estimate_basis(hybrid)
    fit = fit_offsets(hybrid, vx, vy)
    truth_stats = evaluate_truth(truth, fit)

    cv2.imwrite(str(OUT_DIR / "gray_inv.png"), gray_inv)
    cv2.imwrite(str(OUT_DIR / "matched_r5_pos.png"), matched_u8)
    draw_points(raw, truth, hough, OUT_DIR / "hough_overlay.png")
    draw_points(raw, truth, peaks, OUT_DIR / "matched_peaks_overlay.png")
    draw_points(raw, truth, hybrid, OUT_DIR / "hybrid_points_overlay.png")
    draw_grid(marked, truth, fit, OUT_DIR / "hybrid_grid_overlay.png")

    lines = [
        f"truth_count={len(truth)}",
        f"hough_count={len(hough)}",
        f"matched_peak_count={len(peaks)}",
        f"hybrid_count={len(hybrid)}",
        f"origin=({fit.origin[0]:.3f},{fit.origin[1]:.3f})",
        f"vx=({fit.vx[0]:.3f},{fit.vx[1]:.3f}) len={np.linalg.norm(fit.vx):.3f} angle={math.degrees(math.atan2(float(fit.vx[1]), float(fit.vx[0]))):.3f}",
        f"vy=({fit.vy[0]:.3f},{fit.vy[1]:.3f}) len={np.linalg.norm(fit.vy):.3f} angle={math.degrees(math.atan2(float(fit.vy[1]), float(fit.vy[0]))):.3f}",
        f"score={fit.score:.3f}",
        f"good_count={fit.good_count}",
        f"mean_err={fit.mean_err:.4f}",
        f"span_u={fit.span_u}",
        f"span_v={fit.span_v}",
    ]
    for key, value in truth_stats.items():
        lines.append(f"{key}={value:.4f}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
