from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from datamatrix_reader.pipeline import _decode_pure_render, _orient_bits, _render_bits


OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step02_marked_lattice"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_red_centers(marked_bgr: np.ndarray) -> np.ndarray:
    b = marked_bgr[:, :, 0].astype(np.int16)
    g = marked_bgr[:, :, 1].astype(np.int16)
    r = marked_bgr[:, :, 2].astype(np.int16)
    mask = (r > 160) & (r - g > 60) & (r - b > 60)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    points: list[tuple[float, float]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 3 or area > 400:
            continue
        x, y = centroids[idx]
        points.append((float(x), float(y)))
    return np.array(points, dtype=np.float32)


def build_response_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
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
    return {
        "median_dark_31": median_dark_31,
        "matched_r5_pos": matched_r5_pos,
    }


def estimate_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points.astype(np.float32)
    diffs = pts[None, :, :] - pts[:, None, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argpartition(dist, kth=8, axis=1)[:, :8]

    vectors: list[np.ndarray] = []
    for i, nbrs in enumerate(nearest):
        p = pts[i]
        for j in nbrs:
            v = pts[j] - p
            d = float(np.linalg.norm(v))
            if 12.5 <= d <= 22.0:
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


def score_origin(points: np.ndarray, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    basis = np.column_stack((vx, vy))
    inv = np.linalg.inv(basis)
    local = (points - origin[None, :]) @ inv.T
    ij = np.rint(local).astype(np.int32)
    valid = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < 20)
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < 20)
    )
    recon = origin[None, :] + ij[:, 0:1] * vx[None, :] + ij[:, 1:2] * vy[None, :]
    err = np.linalg.norm(points - recon, axis=1)
    good = valid & (err <= 6.5)
    unique = len({(int(a), int(b)) for a, b in ij[good]})
    score = float(good.sum()) + 0.35 * unique - 0.12 * float(np.square(err[good]).mean() if np.any(good) else 30.0)
    return score, ij, err


def fit_lattice(points: np.ndarray, vx_seed: np.ndarray, vy_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    min_x, min_y = points.min(axis=0)
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None = None

    angle_x = math.degrees(math.atan2(float(vx_seed[1]), float(vx_seed[0])))
    angle_y = math.degrees(math.atan2(float(vy_seed[1]), float(vy_seed[0])))
    len_x = float(np.linalg.norm(vx_seed))
    len_y = float(np.linalg.norm(vy_seed))

    for d_ax in np.linspace(-1.25, 1.25, 6):
        for d_ay in np.linspace(-1.25, 1.25, 6):
            ax = math.radians(angle_x + d_ax)
            ay = math.radians(angle_y + d_ay)
            for sx in np.linspace(len_x - 0.8, len_x + 0.8, 5):
                for sy in np.linspace(len_y - 0.8, len_y + 0.8, 5):
                    vx = np.array([math.cos(ax) * sx, math.sin(ax) * sx], dtype=np.float32)
                    vy = np.array([math.cos(ay) * sy, math.sin(ay) * sy], dtype=np.float32)
                    if abs(np.cross(vx, vy)) < 40.0:
                        continue
                    for ox in np.linspace(min_x - 4.0, min_x + 10.0, 15):
                        for oy in np.linspace(min_y - 4.0, min_y + 10.0, 15):
                            origin = np.array([ox, oy], dtype=np.float32)
                            score, ij, err = score_origin(points, origin, vx, vy)
                            meta = {
                                "score": score,
                                "good": float(np.sum(err <= 6.5)),
                                "mean_err": float(err.mean()),
                                "ax": angle_x + d_ax,
                                "ay": angle_y + d_ay,
                                "sx": sx,
                                "sy": sy,
                            }
                            if best is None or score > best[0]:
                                best = (score, origin.copy(), vx.copy(), vy.copy(), meta)

    assert best is not None
    return best[1], best[2], best[3], best[4]


def sample_grid(response: np.ndarray, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray, patch_radius: int = 4) -> np.ndarray:
    scores = np.zeros((20, 20), dtype=np.float32)
    h, w = response.shape[:2]
    for row in range(20):
        for col in range(20):
            pt = origin + col * vx + row * vy
            cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
            x0 = max(0, cx - patch_radius)
            y0 = max(0, cy - patch_radius)
            x1 = min(w, cx + patch_radius + 1)
            y1 = min(h, cy + patch_radius + 1)
            patch = response[y0:y1, x0:x1]
            scores[row, col] = float(patch.max()) if patch.size else 0.0
    return scores


def draw_overlay(image: np.ndarray, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    vis = image.copy()
    for row in range(20):
        for col in range(20):
            pt = origin + col * vx + row * vy
            cv2.circle(vis, (int(round(float(pt[0]))), int(round(float(pt[1])))), 3, color, 1, lineType=cv2.LINE_AA)
    tl = origin
    tr = origin + 19 * vx
    bl = origin + 19 * vy
    br = origin + 19 * vx + 19 * vy
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [quad], True, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    return vis


def save_heatmap(scores: np.ndarray, path: Path) -> None:
    norm = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    big = cv2.resize(norm, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), big)


def render_candidate(scores: np.ndarray, quantile: float) -> tuple[np.ndarray, str | None]:
    threshold = float(np.quantile(scores, quantile))
    bits = (scores >= threshold).astype(np.uint8)
    _, oriented = _orient_bits(bits)
    rendered = _render_bits(oriented)
    text = _decode_pure_render(rendered)
    return rendered, text


def main() -> None:
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    raw = cv2.imread(str(ROOT / "cropped.png"))
    if marked is None or raw is None:
        raise SystemExit("Input images missing.")

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    points = extract_red_centers(marked)
    responses = build_response_maps(gray)
    vx_seed, vy_seed = estimate_basis(points)
    origin, vx, vy, meta = fit_lattice(points, vx_seed, vy_seed)

    raw_overlay = draw_overlay(raw, origin, vx, vy)
    marked_overlay = draw_overlay(marked, origin, vx, vy, color=(0, 255, 0))
    cv2.imwrite(str(OUT_DIR / "marked_grid_overlay.png"), marked_overlay)
    cv2.imwrite(str(OUT_DIR / "raw_grid_overlay.png"), raw_overlay)

    matched_scores = sample_grid(responses["matched_r5_pos"], origin, vx, vy)
    median_scores = sample_grid(responses["median_dark_31"], origin, vx, vy)
    combined_scores = matched_scores - 0.55 * median_scores

    save_heatmap(matched_scores, OUT_DIR / "matched_scores_heatmap.png")
    save_heatmap(median_scores, OUT_DIR / "median_scores_heatmap.png")
    save_heatmap(combined_scores, OUT_DIR / "combined_scores_heatmap.png")

    np.savetxt(OUT_DIR / "combined_scores.csv", combined_scores, delimiter=",", fmt="%.6f")

    summary_lines = [
        f"point_count={len(points)}",
        f"origin=({origin[0]:.3f},{origin[1]:.3f})",
        f"vx=({vx[0]:.3f},{vx[1]:.3f}) len={np.linalg.norm(vx):.3f}",
        f"vy=({vy[0]:.3f},{vy[1]:.3f}) len={np.linalg.norm(vy):.3f}",
        f"score={meta['score']:.3f}",
        f"seed_ax={meta['ax']:.3f}",
        f"seed_ay={meta['ay']:.3f}",
    ]

    decoded_hits: list[str] = []
    for quantile in (0.48, 0.50, 0.52, 0.54, 0.56):
        rendered, text = render_candidate(combined_scores, quantile)
        out_name = f"candidate_q{int(round(quantile * 100)):02d}.png"
        cv2.imwrite(str(OUT_DIR / out_name), rendered)
        summary_lines.append(f"q={quantile:.2f} decode={text!r}")
        if text:
            decoded_hits.append(text)

    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    if decoded_hits:
        print("decoded_hits=" + " | ".join(decoded_hits))


if __name__ == "__main__":
    main()
