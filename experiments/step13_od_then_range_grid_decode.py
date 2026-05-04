from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np
import zxingcpp
from ultralytics import YOLO


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "od_range_decode"
MODULE_COUNT = 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datamatrix_reader import pipeline as dm


@dataclass(frozen=True)
class Detection:
    index: int
    cls: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DecodeAttempt:
    detection: Detection
    crop_path: Path
    decoded: list[str]
    quantile: float | None
    scale: float | None
    frame_origin: tuple[float, float] | None
    frame_vx: tuple[float, float] | None
    frame_vy: tuple[float, float] | None


def load_step07_module():
    module_path = ROOT / "experiments" / "step07_combined_edge_fit.py"
    spec = importlib.util.spec_from_file_location("step07_combined_edge_fit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step07_combined_edge_fit"] = module
    spec.loader.exec_module(module)
    return module


def local_range_response(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)


def decode_bits(bits: np.ndarray) -> list[str]:
    decoded: set[str] = set()
    for rotation in range(4):
        rotated = np.rot90(bits, rotation).astype(np.uint8)
        for candidate in (rotated, 1 - rotated):
            rendered = dm._render_bits(candidate)
            decoded.update(result.text for result in zxingcpp.read_barcodes(rendered))
    return sorted(decoded)


def sample_scores(
    response: np.ndarray,
    origin: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
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


def draw_detection_overlay(
    image: np.ndarray,
    detections: list[Detection],
    path: Path,
) -> None:
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


def draw_grid_overlay(
    image: np.ndarray,
    origin: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    path: Path,
) -> None:
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
    crop: np.ndarray,
    grid_overlay: np.ndarray,
    response: np.ndarray,
    bits: np.ndarray,
    path: Path,
) -> None:
    response_bgr = cv2.applyColorMap(
        cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    panels = [
        labeled_panel(crop, "1 OD crop"),
        labeled_panel(grid_overlay, "2 fitted 20x20 grid"),
        labeled_panel(response_bgr, "3 local range response"),
        labeled_panel(dm._render_bits(bits), "4 rendered bits"),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    cv2.imwrite(str(path), np.hstack(padded))


def detect_datamatrix(
    model_path: Path,
    image_path: Path,
    confidence: float,
) -> tuple[np.ndarray, list[Detection]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    model = YOLO(str(model_path))
    result = model.predict(str(image_path), conf=confidence, verbose=False)[0]
    detections: list[Detection] = []
    if result.boxes is None:
        return image, detections
    for index, box in enumerate(result.boxes):
        xyxy = tuple(float(value) for value in box.xyxy.cpu().numpy()[0])
        detections.append(
            Detection(
                index=index,
                cls=int(box.cls.cpu().numpy()[0]),
                confidence=float(box.conf.cpu().numpy()[0]),
                xyxy=xyxy,
            )
        )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return image, detections


def fit_frame(step07, gray: np.ndarray):
    points, weights, _ = step07.build_weighted_points(gray)
    best = None
    for vx, vy in step07.candidate_bases():
        try:
            fit = step07.fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit
    if best is None:
        raise RuntimeError("No grid frame fit")
    return best


def decode_crop(
    step07,
    crop: np.ndarray,
    stem: str,
    det: Detection,
) -> DecodeAttempt:
    crop_path = OUT_DIR / f"{stem}_det{det.index}_crop.png"
    cv2.imwrite(str(crop_path), crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    response = local_range_response(gray)
    cv2.imwrite(str(OUT_DIR / f"{stem}_det{det.index}_range_response.png"), response)
    fit = fit_frame(step07, gray)

    best_debug: tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for scale in (0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25):
        origin = np.asarray(fit.origin, dtype=np.float32)
        vx = np.asarray(fit.vx, dtype=np.float32) * scale
        vy = np.asarray(fit.vy, dtype=np.float32) * scale
        scores = sample_scores(response, origin, vx, vy)
        for quantile in (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52):
            threshold = float(np.quantile(scores, quantile))
            bits = (scores >= threshold).astype(np.uint8)
            decoded = decode_bits(bits)
            if decoded:
                grid_path = OUT_DIR / f"{stem}_det{det.index}_grid_overlay.png"
                draw_grid_overlay(crop, origin, vx, vy, grid_path)
                grid_overlay = cv2.imread(str(grid_path), cv2.IMREAD_COLOR)
                bits_path = OUT_DIR / f"{stem}_det{det.index}_q{quantile:.2f}_bits.png"
                cv2.imwrite(str(bits_path), dm._render_bits(bits))
                write_pipeline_preview(
                    crop,
                    grid_overlay,
                    response,
                    bits,
                    OUT_DIR / f"{stem}_det{det.index}_pipeline.png",
                )
                return DecodeAttempt(
                    detection=det,
                    crop_path=crop_path,
                    decoded=decoded,
                    quantile=quantile,
                    scale=scale,
                    frame_origin=(float(origin[0]), float(origin[1])),
                    frame_vx=(float(vx[0]), float(vx[1])),
                    frame_vy=(float(vy[0]), float(vy[1])),
                )
            score = float(abs(bits.mean() - 0.5))
            if best_debug is None or score < best_debug[0]:
                best_debug = (score, quantile, bits, origin, vx, vy)

    if best_debug is not None:
        _, quantile, bits, origin, vx, vy = best_debug
        grid_path = OUT_DIR / f"{stem}_det{det.index}_best_grid_overlay.png"
        draw_grid_overlay(crop, origin, vx, vy, grid_path)
        cv2.imwrite(
            str(OUT_DIR / f"{stem}_det{det.index}_best_q{quantile:.2f}_bits.png"),
            dm._render_bits(bits),
        )
    return DecodeAttempt(
        detection=det,
        crop_path=crop_path,
        decoded=[],
        quantile=None,
        scale=None,
        frame_origin=None,
        frame_vx=None,
        frame_vy=None,
    )


def run(
    image_path: Path = ROOT / "saf.png",
    model_path: Path = ROOT / "best.pt",
    confidence: float = 0.5,
    max_detections: int = 3,
) -> list[DecodeAttempt]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step07 = load_step07_module()
    image, detections = detect_datamatrix(model_path, image_path, confidence)
    stem = image_path.stem
    draw_detection_overlay(image, detections, OUT_DIR / f"{stem}_detections_overlay.png")

    attempts: list[DecodeAttempt] = []
    for det in detections[:max_detections]:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2].copy()
        attempt = decode_crop(step07, crop, stem, det)
        attempts.append(attempt)
        if attempt.decoded:
            break

    lines = [
        "od_then_range_grid_decode",
        f"image={image_path.name}",
        f"model={model_path.name}",
        f"detections={len(detections)}",
        "",
    ]
    for attempt in attempts:
        det = attempt.detection
        lines.append(
            (
                f"det{det.index}: conf={det.confidence:.4f} "
                f"xyxy={[round(v, 1) for v in det.xyxy]} "
                f"scale={attempt.scale} q={attempt.quantile} "
                f"decoded={attempt.decoded}"
            )
        )
    summary = "\n".join(lines)
    (OUT_DIR / f"{stem}_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return attempts


def main() -> None:
    run()


if __name__ == "__main__":
    main()
