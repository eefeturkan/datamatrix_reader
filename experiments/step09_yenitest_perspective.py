from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "yenitest_perspective"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_step07():
    path = ROOT / "experiments" / "step07_combined_edge_fit.py"
    spec = importlib.util.spec_from_file_location("step07_combined_edge_fit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = load_step07()
    image = cv2.imread(str(ROOT / "yenitest.png"))
    if image is None:
        raise SystemExit("yenitest.png not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points, weights, _ = mod.build_weighted_points(gray)

    best = None
    for vx, vy in mod.candidate_bases():
        try:
            fit = mod.fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit

    if best is None:
        raise SystemExit("No frame fit found")

    tl = best.origin - 0.5 * best.vx - 0.5 * best.vy
    tr = best.origin + 19.5 * best.vx - 0.5 * best.vy
    bl = best.origin - 0.5 * best.vx + 19.5 * best.vy
    br = best.origin + 19.5 * best.vx + 19.5 * best.vy

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    module_size = 20
    margin = 20
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

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (side + 2 * margin, side + 2 * margin),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    overlay = image.copy()
    quad = np.round(src).astype(np.int32)
    cv2.polylines(overlay, [quad], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)
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

    gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.addWeighted(gray_warp, 1.5, cv2.GaussianBlur(gray_warp, (0, 0), 0.9), -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(sharpened)

    cv2.imwrite(str(OUT_DIR / "yenitest_frame_overlay.png"), overlay)
    cv2.imwrite(str(OUT_DIR / "yenitest_perspective_corrected.png"), warped)
    cv2.imwrite(str(OUT_DIR / "yenitest_perspective_corrected_gray.png"), gray_warp)
    cv2.imwrite(str(OUT_DIR / "yenitest_perspective_corrected_enhanced.png"), clahe)

    summary = "\n".join(
        [
            f"origin=({best.origin[0]:.3f},{best.origin[1]:.3f})",
            f"vx=({best.vx[0]:.3f},{best.vx[1]:.3f})",
            f"vy=({best.vy[0]:.3f},{best.vy[1]:.3f})",
            f"score={best.score:.3f}",
            "corners=" + repr(np.round(src, 3).tolist()),
            f"output_path={OUT_DIR / 'yenitest_perspective_corrected.png'}",
        ]
    )
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
