from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
import zxingcpp
from pylibdmtx.pylibdmtx import encode as encode_datamatrix


@dataclass(frozen=True)
class Candidate:
    name: str
    image: np.ndarray
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True)
class Detection:
    index: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DecodeHit:
    text: str
    candidate: Candidate
    kernel_size: int
    pad: int
    pad_value: int
    scale: int
    inverted: bool
    processed: np.ndarray
    result: object


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or "decoded_barcode"


def normalize_u8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def local_range_response(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def detect_with_yolo(model_path: Path, image_path: Path, conf: float) -> list[Detection]:
    if not model_path.exists():
        return []
    try:
        from ultralytics import YOLO
    except Exception:
        return []

    model = YOLO(str(model_path), task="detect")
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]
    if result.boxes is None:
        return []

    detections: list[Detection] = []
    for index, box in enumerate(result.boxes):
        detections.append(
            Detection(
                index=index,
                confidence=float(box.conf.cpu().numpy()[0]),
                xyxy=tuple(float(value) for value in box.xyxy.cpu().numpy()[0]),
            )
        )
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def crop_with_padding(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    padding: int,
    name: str,
) -> Candidate:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return Candidate(name=name, image=image[y1:y2, x1:x2].copy(), offset_x=x1, offset_y=y1)


def build_candidates(
    image: np.ndarray,
    image_path: Path,
    model_path: Path,
    conf: float,
    max_detections: int,
    skip_od: bool,
) -> tuple[list[Candidate], list[Detection]]:
    candidates = [Candidate(name="full_image", image=image)]
    detections: list[Detection] = []
    if skip_od:
        return candidates, detections

    detections = detect_with_yolo(model_path, image_path, conf)
    for det in detections[:max_detections]:
        candidates.append(crop_with_padding(image, det.xyxy, 80, f"det{det.index}_pad80"))
        candidates.append(crop_with_padding(image, det.xyxy, 140, f"det{det.index}_pad140"))
    return candidates, detections


def variant_params() -> list[tuple[int, int, int, int]]:
    preferred = [
        (9, 0, 0, 2),
        (9, 50, 0, 1),
        (9, 0, 0, 1),
        (7, 50, 0, 1),
        (7, 0, 0, 2),
        (5, 0, 0, 2),
        (3, 0, 0, 1),
        (9, 50, 0, 2),
        (7, 50, 0, 2),
        (11, 0, 0, 2),
    ]
    all_params: list[tuple[int, int, int, int]] = []
    for kernel_size in (9, 7, 5, 3, 11):
        for pad in (0, 50, 100):
            pad_values = (0,) if pad == 0 else (0, 255)
            for pad_value in pad_values:
                for scale in (2, 1, 3, 4):
                    all_params.append((kernel_size, pad, pad_value, scale))

    ordered: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for params in preferred + all_params:
        if params in seen:
            continue
        seen.add(params)
        ordered.append(params)
    return ordered


def processed_variants(
    gray: np.ndarray,
    max_variants: int | None = None,
) -> Iterable[tuple[int, int, int, int, bool, np.ndarray]]:
    emitted = 0
    for kernel_size, pad, pad_value, scale in variant_params():
        if max_variants is not None and emitted >= max_variants:
            return
        response = local_range_response(gray, kernel_size)
        padded = (
            cv2.copyMakeBorder(
                response,
                pad,
                pad,
                pad,
                pad,
                cv2.BORDER_CONSTANT,
                value=pad_value,
            )
            if pad
            else response
        )
        scaled = (
            cv2.resize(
                padded,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            if scale > 1
            else padded
        )
        yield kernel_size, pad, pad_value, scale, False, scaled
        emitted += 1
        if max_variants is not None and emitted >= max_variants:
            return
        yield kernel_size, pad, pad_value, scale, True, 255 - scaled
        emitted += 1


def decode_candidate(candidate: Candidate, max_variants: int | None = None) -> DecodeHit | None:
    gray = cv2.cvtColor(candidate.image, cv2.COLOR_BGR2GRAY)
    for kernel_size, pad, pad_value, scale, inverted, processed in processed_variants(gray, max_variants):
        results = zxingcpp.read_barcodes(processed)
        for result in results:
            if str(result.format) == "Data Matrix" and result.text:
                return DecodeHit(
                    text=result.text,
                    candidate=candidate,
                    kernel_size=kernel_size,
                    pad=pad,
                    pad_value=pad_value,
                    scale=scale,
                    inverted=inverted,
                    processed=processed,
                    result=result,
                )
    return None


def result_points_in_original(hit: DecodeHit) -> np.ndarray | None:
    position = getattr(hit.result, "position", None)
    if position is None:
        return None

    pts = []
    for point in (
        position.top_left,
        position.top_right,
        position.bottom_right,
        position.bottom_left,
    ):
        x = (float(point.x) / hit.scale) - hit.pad + hit.candidate.offset_x
        y = (float(point.y) / hit.scale) - hit.pad + hit.candidate.offset_y
        pts.append((x, y))
    return np.array(pts, dtype=np.float32)


def crop_from_points(image: np.ndarray, points: np.ndarray | None, fallback: Candidate, margin: int) -> np.ndarray:
    if points is None:
        return fallback.image.copy()
    h, w = image.shape[:2]
    x1 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y1 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x2 = min(w, int(np.ceil(points[:, 0].max())) + margin)
    y2 = min(h, int(np.ceil(points[:, 1].max())) + margin)
    if x2 <= x1 or y2 <= y1:
        return fallback.image.copy()
    return image[y1:y2, x1:x2].copy()


def draw_detections(image: np.ndarray, detections: list[Detection], path: Path) -> None:
    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            overlay,
            f"{det.index}:{det.confidence:.3f}",
            (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), overlay)


def draw_position(image: np.ndarray, points: np.ndarray | None, path: Path) -> None:
    overlay = image.copy()
    if points is not None:
        cv2.polylines(
            overlay,
            [np.round(points).astype(np.int32)],
            True,
            (0, 255, 255),
            4,
            lineType=cv2.LINE_AA,
        )
        for point in points:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 6, (0, 0, 255), -1)
    cv2.imwrite(str(path), overlay)


def labeled_panel(image: np.ndarray, label: str, width: int = 360) -> np.ndarray:
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


def write_pipeline_preview(hit: DecodeHit, crop: np.ndarray, points_overlay: np.ndarray, path: Path) -> None:
    gray_crop = cv2.cvtColor(hit.candidate.image, cv2.COLOR_BGR2GRAY)
    response = local_range_response(gray_crop, hit.kernel_size)
    response_color = cv2.applyColorMap(normalize_u8(response), cv2.COLORMAP_TURBO)
    panels = [
        labeled_panel(hit.candidate.image, "1 candidate crop"),
        labeled_panel(points_overlay, "2 decoded position"),
        labeled_panel(response_color, f"3 local range k={hit.kernel_size}"),
        labeled_panel(hit.processed, "4 zxing input"),
        labeled_panel(crop, "5 barcode crop"),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    cv2.imwrite(str(path), np.hstack(padded))


def write_decoded_datamatrix(text: str, path: Path) -> None:
    encoded = encode_datamatrix(text.encode("utf-8"))
    image = np.frombuffer(encoded.pixels, dtype=np.uint8).reshape(
        encoded.height,
        encoded.width,
        encoded.bpp // 8,
    )
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), image)


def run(
    image_path: Path,
    model_path: Path,
    output_dir: Path,
    conf: float,
    max_detections: int,
    skip_od: bool,
    full_image_variants: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)
    detections: list[Detection] = []
    candidates = [Candidate(name="full_image", image=image)]

    for candidate in candidates:
        hit = decode_candidate(candidate, max_variants=None if skip_od else full_image_variants)
        if hit is None:
            print(f"{candidate.name}: no decode")
            continue
        return write_success(image, image_path, output_dir, hit, detections)

    if not skip_od:
        candidates, detections = build_candidates(
            image=image,
            image_path=image_path,
            model_path=model_path,
            conf=conf,
            max_detections=max_detections,
            skip_od=False,
        )
        draw_detections(image, detections, output_dir / f"{image_path.stem}_detections.png")

        for candidate in candidates[1:]:
            hit = decode_candidate(candidate)
            if hit is None:
                print(f"{candidate.name}: no decode")
                continue
            return write_success(image, image_path, output_dir, hit, detections)

    draw_detections(image, detections, output_dir / f"{image_path.stem}_detections.png")
    print("No decoded Data Matrix")
    return 2


def write_success(
    image: np.ndarray,
    image_path: Path,
    output_dir: Path,
    hit: DecodeHit,
    detections: list[Detection],
) -> int:
    draw_detections(image, detections, output_dir / f"{image_path.stem}_detections.png")
    name = safe_name(hit.text)
    points = result_points_in_original(hit)
    barcode_crop = crop_from_points(image, points, hit.candidate, margin=35)

    crop_path = output_dir / f"{name}_crop.png"
    processed_path = output_dir / f"{name}_dark_range_zxing_input.png"
    decoded_matrix_path = output_dir / f"{name}_decoded_datamatrix.png"
    position_path = output_dir / f"{name}_position_overlay.png"
    pipeline_path = output_dir / f"{name}_pipeline.png"
    metadata_path = output_dir / f"{name}_metadata.txt"

    cv2.imwrite(str(crop_path), barcode_crop)
    cv2.imwrite(str(processed_path), hit.processed)
    write_decoded_datamatrix(hit.text, decoded_matrix_path)
    draw_position(image, points, position_path)
    points_overlay = cv2.imread(str(position_path), cv2.IMREAD_COLOR)
    write_pipeline_preview(hit, barcode_crop, points_overlay, pipeline_path)

    metadata_path.write_text(
        "\n".join(
            [
                f"decoded_text={hit.text}",
                f"source_image={image_path}",
                f"candidate={hit.candidate.name}",
                f"candidate_offset=({hit.candidate.offset_x},{hit.candidate.offset_y})",
                f"kernel_size={hit.kernel_size}",
                f"pad={hit.pad}",
                f"pad_value={hit.pad_value}",
                f"scale={hit.scale}",
                f"inverted={hit.inverted}",
                f"format={hit.result.format}",
                f"position={hit.result.position}",
                f"decoded_datamatrix={decoded_matrix_path}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"decoded: {hit.text}")
    print(f"candidate: {hit.candidate.name}")
    print(
        "preprocess: "
        f"local_range k={hit.kernel_size}, pad={hit.pad}, "
        f"pad_value={hit.pad_value}, scale={hit.scale}, inverted={hit.inverted}"
    )
    print(f"crop: {crop_path}")
    print(f"processed: {processed_path}")
    print(f"decoded datamatrix: {decoded_matrix_path}")
    print(f"pipeline: {pipeline_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dark-surface Data Matrix reader using local-range preprocessing and ZXing."
    )
    parser.add_argument("--image", type=Path, required=True, help="Raw image path")
    parser.add_argument("--model", type=Path, default=Path("best.onnx"), help="Optional YOLO model path")
    parser.add_argument("--out", type=Path, default=Path("artifacts/dark_surface_reader"))
    parser.add_argument("--conf", type=float, default=0.01, help="Low OD confidence for dark-surface fallback")
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--full-image-variants", type=int, default=24)
    parser.add_argument("--skip-od", action="store_true", help="Use full-image preprocessing only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run(
            image_path=args.image,
            model_path=args.model,
            output_dir=args.out,
            conf=args.conf,
            max_detections=args.max_detections,
            skip_od=args.skip_od,
            full_image_variants=args.full_image_variants,
        )
    )


if __name__ == "__main__":
    main()
