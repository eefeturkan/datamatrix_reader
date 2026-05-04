from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import csv

import cv2
import numpy as np
import zxingcpp

ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "range_grid_decode"
MODULE_COUNT = 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datamatrix_reader import pipeline as dm


@dataclass(frozen=True)
class GridFrame:
    origin: tuple[float, float]
    vx: tuple[float, float]
    vy: tuple[float, float]


# Current focus-image calibrations. In the factory setup this should come from
# a one-time calibration or a small Hough/lattice search, then stay stable.
GRID_FRAMES = {
    "cropped.png": GridFrame(
        origin=(91.955, 51.579),
        vx=(15.765, 1.047),
        vy=(-1.394, 15.939),
    ),
    "yenitest.png": GridFrame(
        origin=(20.0, 26.5),
        vx=(24.0, 0.0),
        vy=(0.0, 24.0),
    ),
    "yenitest2.png": GridFrame(
        origin=(40.0, 22.0),
        vx=(19.7368421053, 1.4736842105),
        vy=(-1.1578947368, 20.0),
    ),
}


def local_range_response(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)


def sample_cell_scores(response: np.ndarray, frame: GridFrame) -> np.ndarray:
    origin = np.asarray(frame.origin, dtype=np.float32)
    vx = np.asarray(frame.vx, dtype=np.float32)
    vy = np.asarray(frame.vy, dtype=np.float32)
    scores = np.zeros((MODULE_COUNT, MODULE_COUNT), dtype=np.float32)
    for row in range(MODULE_COUNT):
        for col in range(MODULE_COUNT):
            point = origin + col * vx + row * vy
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            patch = response[
                max(0, y - 6) : min(response.shape[0], y + 7),
                max(0, x - 6) : min(response.shape[1], x + 7),
            ]
            scores[row, col] = float(np.percentile(patch, 90)) if patch.size else 0.0
    return scores


def decode_bits(bits: np.ndarray) -> list[str]:
    decoded: set[str] = set()
    for rotation in range(4):
        rotated = np.rot90(bits, rotation).astype(np.uint8)
        for candidate in (rotated, 1 - rotated):
            rendered = dm._render_bits(candidate)
            decoded.update(result.text for result in zxingcpp.read_barcodes(rendered))
    return sorted(decoded)


def draw_grid_overlay(image: np.ndarray, frame: GridFrame, path: Path) -> None:
    origin = np.asarray(frame.origin, dtype=np.float32)
    vx = np.asarray(frame.vx, dtype=np.float32)
    vy = np.asarray(frame.vy, dtype=np.float32)
    overlay = image.copy()
    corners = np.array(
        [
            origin,
            origin + (MODULE_COUNT - 1) * vx,
            origin + (MODULE_COUNT - 1) * vx + (MODULE_COUNT - 1) * vy,
            origin + (MODULE_COUNT - 1) * vy,
        ],
        dtype=np.float32,
    )
    cv2.polylines(
        overlay,
        [np.round(corners).astype(np.int32)],
        True,
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    for row in range(MODULE_COUNT):
        for col in range(MODULE_COUNT):
            point = origin + col * vx + row * vy
            cv2.circle(
                overlay,
                tuple(np.round(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), overlay)


def labeled_panel(image: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    bar = np.full((32, resized.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, resized])


def write_pipeline_preview(
    image_name: str,
    image: np.ndarray,
    overlay_path: Path,
    response: np.ndarray,
    bits_path: Path | None,
) -> None:
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    response_bgr = cv2.applyColorMap(
        cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    panels = [
        labeled_panel(image, "1 original"),
        labeled_panel(overlay, "2 calibrated 20x20 grid"),
        labeled_panel(response_bgr, "3 local range response"),
    ]
    if bits_path is not None:
        bits = cv2.imread(str(bits_path), cv2.IMREAD_GRAYSCALE)
        panels.append(labeled_panel(bits, "4 rendered bits"))
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    preview = np.hstack(padded)
    cv2.imwrite(str(OUT_DIR / f"{Path(image_name).stem}_pipeline.png"), preview)


def decode_image(image_name: str) -> tuple[str, list[str], list[dict[str, str]]]:
    image_path = ROOT / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    frame = GRID_FRAMES[image_name]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = local_range_response(gray)
    scores = sample_cell_scores(response, frame)

    stem = Path(image_name).stem
    cv2.imwrite(str(OUT_DIR / f"{stem}_range_response.png"), response)
    overlay_path = OUT_DIR / f"{stem}_grid_overlay.png"
    draw_grid_overlay(image, frame, overlay_path)

    lines: list[str] = []
    rows: list[dict[str, str]] = []
    first_decoded_bits_path: Path | None = None
    for quantile in (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52):
        threshold = float(np.quantile(scores, quantile))
        bits = (scores >= threshold).astype(np.uint8)
        decoded = decode_bits(bits)
        if decoded:
            rendered = dm._render_bits(bits)
            bits_path = OUT_DIR / f"{stem}_q{quantile:.2f}_bits.png"
            cv2.imwrite(str(bits_path), rendered)
            if first_decoded_bits_path is None:
                first_decoded_bits_path = bits_path
        lines.append(
            f"q={quantile:.2f} occupancy={bits.mean():.3f} decoded={decoded}"
        )
        rows.append(
            {
                "image": image_name,
                "quantile": f"{quantile:.2f}",
                "occupancy": f"{bits.mean():.3f}",
                "decoded": " | ".join(decoded),
            }
        )
    write_pipeline_preview(image_name, image, overlay_path, response, first_decoded_bits_path)
    return image_name, lines, rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "range_grid_decode",
        "method=20x20 calibrated grid + 7x7 local range response + 13x13 cell patch p90 + quantile bit threshold",
        "",
    ]
    csv_rows: list[dict[str, str]] = []
    for image_name in ("cropped.png", "yenitest.png", "yenitest2.png"):
        name, lines, rows = decode_image(image_name)
        csv_rows.extend(rows)
        summary_lines.append(name)
        summary_lines.extend(f"  {line}" for line in lines)
        summary_lines.append("")
    summary = "\n".join(summary_lines)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "quantile", "occupancy", "decoded"))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(summary)


if __name__ == "__main__":
    main()
