from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
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


def local_contrast_normalize(gray: np.ndarray, win: int = 41, eps: float = 1e-3) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(gray_f, (0, 0), win / 6.0)
    sq_mu = cv2.GaussianBlur(gray_f * gray_f, (0, 0), win / 6.0)
    sigma = np.sqrt(np.maximum(sq_mu - mu * mu, 0.0))
    return (gray_f - mu) / (sigma + eps)


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
        isotropy = np.minimum(np.abs(l1), np.abs(l2)) / (
            np.maximum(np.abs(l1), np.abs(l2)) + 1e-6
        )
        blobness = np.where(same_sign, np.sqrt(np.maximum(det, 0.0)) * isotropy, 0.0)
        maps.append(normalize_u8(blobness))
    return np.max(np.stack(maps, axis=0), axis=0).astype(np.uint8)


def fft_best_map(dark_lcn: np.ndarray) -> np.ndarray:
    h, w = dark_lcn.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)
    f0 = 1.0 / 17.0
    width = 0.014
    mask = np.exp(-0.5 * ((radius - f0) / width) ** 2)
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


def merge_weighted_points(
    points: np.ndarray, weights: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
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


def build_weighted_points(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lcn = local_contrast_normalize(gray, win=41)
    dark = np.clip(-lcn, 0.0, None)
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
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return merge_weighted_points(
        np.array(points, dtype=np.float32),
        np.array(weights, dtype=np.float32),
        radius=5.5,
    )


def candidate_bases() -> list[tuple[np.ndarray, np.ndarray]]:
    bases: list[tuple[np.ndarray, np.ndarray]] = []
    for ax in (3.0, 3.8, 4.6):
        for ay in (95.0, 96.5, 98.0):
            for sx in (15.8, 16.2):
                for sy in (16.0, 16.6):
                    vx = np.array(
                        [math.cos(math.radians(ax)) * sx, math.sin(math.radians(ax)) * sx],
                        dtype=np.float32,
                    )
                    vy = np.array(
                        [math.cos(math.radians(ay)) * sy, math.sin(math.radians(ay)) * sy],
                        dtype=np.float32,
                    )
                bases.append((vx, vy))
    return bases


def fit_edges(points: np.ndarray, weights: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> GridFit:
    basis = np.column_stack((vx, vy))
    inv = np.linalg.inv(basis)
    coords = points @ inv.T
    best: GridFit | None = None

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
                    fit = GridFit(score, origin.astype(np.float32), vx.copy(), vy.copy())
                    if best is None or fit.score > best.score:
                        best = fit
    if best is None:
        raise RuntimeError("No edge fit found")
    return best


def fit_grid_frame(points: np.ndarray, weights: np.ndarray) -> GridFit:
    if len(points) == 0:
        raise RuntimeError("No weighted points for grid fit")
    best: GridFit | None = None
    for vx, vy in candidate_bases():
        try:
            fit = fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit
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
