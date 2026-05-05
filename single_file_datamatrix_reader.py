from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import zxingcpp
from ultralytics import YOLO


MODULE_COUNT = 20


@dataclass(frozen=True)
class Detection:
    index: int
    cls: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class GridFit:
    score: float
    origin: np.ndarray
    vx: np.ndarray
    vy: np.ndarray


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or "decoded_barcode"


def local_range_response(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)


def render_bits(bits: np.ndarray) -> np.ndarray:
    canvas = 255 * (1 - bits.astype(np.uint8))
    image = np.kron(canvas, np.ones((16, 16), dtype=np.uint8))
    return cv2.copyMakeBorder(image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)


def decode_bits(bits: np.ndarray) -> list[str]:
    decoded: set[str] = set()
    for rotation in range(4):
        rotated = np.rot90(bits, rotation).astype(np.uint8)
        for candidate in (rotated, 1 - rotated):
            rendered = render_bits(candidate)
            decoded.update(result.text for result in zxingcpp.read_barcodes(rendered))
    return sorted(decoded)


def detect_datamatrix(model_path: Path, image_path: Path, conf: float) -> tuple[np.ndarray, list[Detection]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = YOLO(str(model_path), task="detect")
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    detections: list[Detection] = []
    if result.boxes is None:
        return image, detections
    for index, box in enumerate(result.boxes):
        detections.append(
            Detection(
                index=index,
                cls=int(box.cls.cpu().numpy()[0]),
                confidence=float(box.conf.cpu().numpy()[0]),
                xyxy=tuple(float(value) for value in box.xyxy.cpu().numpy()[0]),
            )
        )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return image, detections


def normalize_u8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def build_weighted_points(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    response = local_range_response(gray)
    response = cv2.GaussianBlur(response, (0, 0), 0.8)
    threshold = float(np.percentile(response, 86))
    mask = (response >= threshold).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > 260:
            continue
        x, y = centers[label]
        points.append((float(x), float(y)))
        weights.append(float(response[labels == label].mean()) * max(1.0, area**0.5))
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.asarray(points, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def candidate_bases(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    bases: list[tuple[np.ndarray, np.ndarray]] = []
    # Estimate dominant pitch from nearest-neighbor distances, then search around it.
    pitch = 18.0
    if len(points) >= 20:
        sample = points[: min(len(points), 400)]
        dists: list[float] = []
        for point in sample:
            delta = sample - point
            lens = np.linalg.norm(delta, axis=1)
            near = lens[(lens > 8.0) & (lens < 35.0)]
            if near.size:
                dists.append(float(np.min(near)))
        if dists:
            pitch = float(np.median(dists))
    for p in np.arange(max(12.0, pitch - 4.0), pitch + 4.01, 0.6):
        for angle_deg in np.arange(-8.0, 8.01, 2.0):
            angle = np.deg2rad(angle_deg)
            vx = np.array([np.cos(angle) * p, np.sin(angle) * p], dtype=np.float32)
            vy_angle = angle + np.pi / 2.0
            for shear in (-1.2, -0.6, 0.0, 0.6, 1.2):
                vy = np.array(
                    [np.cos(vy_angle) * p + shear, np.sin(vy_angle) * p],
                    dtype=np.float32,
                )
                bases.append((vx, vy))
    return bases


def fit_grid_frame(points: np.ndarray, weights: np.ndarray) -> GridFit:
    if len(points) == 0:
        raise RuntimeError("No weighted points for grid fit")
    best: GridFit | None = None
    for vx, vy in candidate_bases(points):
        basis = np.column_stack((vx, vy))
        det = float(np.linalg.det(basis))
        if abs(det) < 1e-3:
            continue
        inv = np.linalg.inv(basis)
        coords = points @ inv.T
        u = coords[:, 0]
        v = coords[:, 1]
        for left in np.percentile(u, [2, 5, 8, 12]):
            for top in np.percentile(v, [2, 5, 8, 12]):
                origin = np.array([left, top], dtype=np.float32) @ basis.T
                rel = (points - origin) @ inv.T
                col = np.rint(rel[:, 0])
                row = np.rint(rel[:, 1])
                du = np.abs(rel[:, 0] - col)
                dv = np.abs(rel[:, 1] - row)
                inside = (
                    (col >= 0)
                    & (col < MODULE_COUNT)
                    & (row >= 0)
                    & (row < MODULE_COUNT)
                    & (du < 0.34)
                    & (dv < 0.34)
                )
                if not np.any(inside):
                    continue
                edge = inside & (
                    (col == 0)
                    | (row == 0)
                    | (col == MODULE_COUNT - 1)
                    | (row == MODULE_COUNT - 1)
                )
                outside = (
                    (col < -1)
                    | (col > MODULE_COUNT)
                    | (row < -1)
                    | (row > MODULE_COUNT)
                )
                score = (
                    1.0 * float(weights[inside].sum())
                    + 4.0 * float(weights[edge].sum())
                    - 0.4 * float(weights[outside].sum())
                    - abs(float(np.mean(inside)) - 0.45) * 50.0
                )
                if best is None or score > best.score:
                    best = GridFit(score=score, origin=origin, vx=vx.copy(), vy=vy.copy())
    if best is None:
        raise RuntimeError("No grid fit found")
    return best


def sample_scores(response: np.ndarray, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
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


def draw_detections(image: np.ndarray, detections: list[Detection], path: Path) -> None:
    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            overlay,
            f"{det.index}:{det.confidence:.2f}",
            (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), overlay)


def draw_grid(crop: np.ndarray, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray, path: Path) -> None:
    overlay = crop.copy()
    corners = np.array(
        [
            origin,
            origin + (MODULE_COUNT - 1) * vx,
            origin + (MODULE_COUNT - 1) * vx + (MODULE_COUNT - 1) * vy,
            origin + (MODULE_COUNT - 1) * vy,
        ],
        dtype=np.float32,
    )
    cv2.polylines(overlay, [np.round(corners).astype(np.int32)], True, (0, 255, 255), 2)
    for row in range(MODULE_COUNT):
        for col in range(MODULE_COUNT):
            point = origin + col * vx + row * vy
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 2, (0, 0, 255), -1)
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
    cv2.putText(bar, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return np.vstack([bar, resized])


def write_pipeline(crop: np.ndarray, grid: np.ndarray, response: np.ndarray, bits: np.ndarray, path: Path) -> None:
    response_bgr = cv2.applyColorMap(normalize_u8(response), cv2.COLORMAP_TURBO)
    panels = [
        labeled_panel(crop, "1 OD crop"),
        labeled_panel(grid, "2 fitted 20x20 grid"),
        labeled_panel(response_bgr, "3 local range response"),
        labeled_panel(render_bits(bits), "4 rendered bits"),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    cv2.imwrite(str(path), np.hstack(padded))


def try_decode_crop(crop: np.ndarray) -> tuple[list[str], float, float, GridFit, np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    points, weights = build_weighted_points(gray)
    fit = fit_grid_frame(points, weights)
    response = local_range_response(gray)
    # The edge fit usually finds the correct lattice direction, but the first
    # grid origin can be off by a few pixels. Search a small local neighborhood
    # before giving up; this keeps the CLI robust without requiring any other
    # project file.
    for scale in (0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25):
        vx = fit.vx * scale
        vy = fit.vy * scale
        for dx in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0):
            for dy in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0):
                origin = fit.origin + np.array([dx, dy], dtype=np.float32)
                scores = sample_scores(response, origin, vx, vy)
                for quantile in (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52):
                    threshold = float(np.quantile(scores, quantile))
                    bits = (scores >= threshold).astype(np.uint8)
                    decoded = decode_bits(bits)
                    if decoded:
                        scaled_fit = GridFit(fit.score, origin, vx, vy)
                        return decoded, scale, quantile, scaled_fit, bits
    return [], 0.0, 0.0, fit, np.zeros((MODULE_COUNT, MODULE_COUNT), dtype=np.uint8)


def run(image_path: Path, model_path: Path, output_dir: Path, conf: float, max_detections: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image, detections = detect_datamatrix(model_path, image_path, conf)
    draw_detections(image, detections, output_dir / f"{image_path.stem}_detections.png")
    if not detections:
        print("No detections")
        return 2

    for det in detections[:max_detections]:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2].copy()
        try:
            decoded, scale, quantile, fit, bits = try_decode_crop(crop)
        except Exception as exc:
            print(f"det{det.index}: failed: {exc}")
            continue
        if not decoded:
            print(f"det{det.index}: no decode")
            continue

        text = decoded[0]
        name = safe_name(text)
        response = local_range_response(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        crop_path = output_dir / f"{name}_crop.png"
        grid_path = output_dir / f"{name}_grid.png"
        response_path = output_dir / f"{name}_range_response.png"
        bits_path = output_dir / f"{name}_bits.png"
        pipeline_path = output_dir / f"{name}_pipeline.png"
        metadata_path = output_dir / f"{name}_metadata.txt"

        cv2.imwrite(str(crop_path), crop)
        draw_grid(crop, fit.origin, fit.vx, fit.vy, grid_path)
        cv2.imwrite(str(response_path), response)
        cv2.imwrite(str(bits_path), render_bits(bits))
        grid = cv2.imread(str(grid_path), cv2.IMREAD_COLOR)
        write_pipeline(crop, grid, response, bits, pipeline_path)
        metadata = "\n".join(
            [
                "single_file_datamatrix_reader",
                f"image={image_path}",
                f"model={model_path}",
                f"decoded_text={text}",
                f"detection_index={det.index}",
                f"detection_confidence={det.confidence:.4f}",
                f"detection_xyxy={[round(v, 1) for v in det.xyxy]}",
                f"scale={scale}",
                f"quantile={quantile}",
                f"frame_origin={[round(float(v), 3) for v in fit.origin]}",
                f"frame_vx={[round(float(v), 3) for v in fit.vx]}",
                f"frame_vy={[round(float(v), 3) for v in fit.vy]}",
            ]
        )
        metadata_path.write_text(metadata, encoding="utf-8")
        print(metadata)
        return 0

    print("No decoded Data Matrix")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file OD + Data Matrix reader")
    parser.add_argument("--image", required=True, type=Path, help="Input full image path")
    parser.add_argument("--model", default=Path("best.onnx"), type=Path, help="YOLO .onnx model path")
    parser.add_argument("--out", default=Path("artifacts/single_file_output"), type=Path)
    parser.add_argument("--conf", default=0.15, type=float, help="YOLO confidence threshold")
    parser.add_argument("--max-detections", default=3, type=int, help="Max boxes to try")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(run(args.image, args.model, args.out, args.conf, args.max_detections))


if __name__ == "__main__":
    main()
