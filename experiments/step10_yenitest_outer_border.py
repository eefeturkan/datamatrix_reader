from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "yenitest_outer_border"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_step07():
    path = ROOT / "experiments" / "step07_combined_edge_fit.py"
    spec = importlib.util.spec_from_file_location("step07_combined_edge_fit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weighted_mode_extreme(values: np.ndarray, weights: np.ndarray, side: str, span: int = 10) -> float:
    rounded = np.rint(values).astype(np.int32)
    min_bin = int(rounded.min())
    max_bin = int(rounded.max())
    hist = {}
    for value, weight in zip(rounded, weights):
        hist[value] = hist.get(value, 0.0) + float(weight)
    if side == "low":
        candidates = list(range(min_bin, min_bin + span + 1))
    else:
        candidates = list(range(max_bin - span, max_bin + 1))
    best_bin = max(candidates, key=lambda idx: hist.get(idx, 0.0))
    mask = np.abs(values - best_bin) <= 0.75
    if np.any(mask):
        return float(np.average(values[mask], weights=weights[mask]))
    return float(best_bin)


def refine_band(values: np.ndarray, weights: np.ndarray, center: float, half_width: float = 0.9) -> float:
    mask = np.abs(values - center) <= half_width
    if np.any(mask):
        return float(np.average(values[mask], weights=weights[mask]))
    return center


def main() -> None:
    mod = load_step07()
    image = cv2.imread(str(ROOT / "yenitest.png"))
    if image is None:
        raise SystemExit("yenitest.png not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points, weights, _ = mod.build_weighted_points(gray)
    if len(points) == 0:
        raise SystemExit("No points found")

    best = None
    for vx, vy in mod.candidate_bases():
        try:
            fit = mod.fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit

    if best is None:
        raise SystemExit("No base fit found")

    basis = np.column_stack((best.vx, best.vy))
    inv = np.linalg.inv(basis)
    coords = points @ inv.T
    u = coords[:, 0]
    v = coords[:, 1]

    left = refine_band(u, weights, weighted_mode_extreme(u, weights, "low"))
    right = refine_band(u, weights, weighted_mode_extreme(u, weights, "high"))
    top = refine_band(v, weights, weighted_mode_extreme(v, weights, "low"))
    bottom = refine_band(v, weights, weighted_mode_extreme(v, weights, "high"))

    def quad(expand: float) -> np.ndarray:
        tl = np.array([left - expand, top - expand], dtype=np.float32) @ basis.T
        tr = np.array([right + expand, top - expand], dtype=np.float32) @ basis.T
        br = np.array([right + expand, bottom + expand], dtype=np.float32) @ basis.T
        bl = np.array([left - expand, bottom + expand], dtype=np.float32) @ basis.T
        return np.array([tl, tr, br, bl], dtype=np.float32)

    src = quad(0.55)
    src_expanded = quad(1.15)

    module_size = 20
    margin = 28
    side = 20 * module_size
    dst = np.array(
        [
            [margin, margin],
            [margin + side, margin],
            [margin + side, margin + side],
            [margin, margin + side],
        ],
        dtype=np.float32,
    )

    def warp_from(src_quad: np.ndarray) -> np.ndarray:
        matrix = cv2.getPerspectiveTransform(src_quad, dst)
        return cv2.warpPerspective(
            image,
            matrix,
            (side + 2 * margin, side + 2 * margin),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    warped = warp_from(src)
    warped_expanded = warp_from(src_expanded)

    overlay = image.copy()
    for point in points:
        cv2.circle(overlay, (int(round(float(point[0]))), int(round(float(point[1])))), 2, (80, 180, 80), -1, lineType=cv2.LINE_AA)
    quad = np.round(src).astype(np.int32)
    cv2.polylines(overlay, [quad], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    quad_expanded = np.round(src_expanded).astype(np.int32)
    cv2.polylines(overlay, [quad_expanded], True, (255, 180, 0), 1, lineType=cv2.LINE_AA)
    for idx, point in enumerate(src):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            ["TL", "TR", "BR", "BL"][idx],
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
        cv2.addWeighted(warped_gray, 1.5, cv2.GaussianBlur(warped_gray, (0, 0), 0.9), -0.5, 0)
    )
    warped_expanded_gray = cv2.cvtColor(warped_expanded, cv2.COLOR_BGR2GRAY)
    warped_expanded_enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
        cv2.addWeighted(warped_expanded_gray, 1.5, cv2.GaussianBlur(warped_expanded_gray, (0, 0), 0.9), -0.5, 0)
    )

    cv2.imwrite(str(OUT_DIR / "yenitest_outer_border_overlay.png"), overlay)
    cv2.imwrite(str(OUT_DIR / "yenitest_outer_border_corrected.png"), warped)
    cv2.imwrite(str(OUT_DIR / "yenitest_outer_border_corrected_enhanced.png"), warped_enhanced)
    cv2.imwrite(str(OUT_DIR / "yenitest_outer_border_corrected_expanded.png"), warped_expanded)
    cv2.imwrite(str(OUT_DIR / "yenitest_outer_border_corrected_expanded_enhanced.png"), warped_expanded_enhanced)

    summary = "\n".join(
        [
            f"left={left:.3f}",
            f"right={right:.3f}",
            f"top={top:.3f}",
            f"bottom={bottom:.3f}",
            f"vx=({best.vx[0]:.3f},{best.vx[1]:.3f})",
            f"vy=({best.vy[0]:.3f},{best.vy[1]:.3f})",
            "corners=" + repr(np.round(src, 3).tolist()),
            "corners_expanded=" + repr(np.round(src_expanded, 3).tolist()),
        ]
    )
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
