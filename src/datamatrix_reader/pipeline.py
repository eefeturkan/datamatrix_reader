from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode as dmtx_decode
import zxingcpp


@dataclass(slots=True)
class CandidateResult:
    text: str | None
    engine: str | None
    stage_name: str
    score: float


@dataclass(slots=True)
class DecodeResult:
    text: str | None
    engine: str | None
    stage_name: str
    score: float
    processed_image: np.ndarray
    alternatives: list[CandidateResult]


@dataclass(slots=True)
class _ImageCandidate:
    stage_name: str
    image: np.ndarray
    preview_score: float
    structural_score: float


@dataclass(slots=True)
class _DecodeHit:
    text: str
    engine: str
    stage_name: str
    valid: bool
    readable_ratio: float
    preview_score: float
    structural_score: float


def decode_image(image_bytes: bytes) -> DecodeResult:
    bgr = _load_image(image_bytes)
    raw_grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grayscale = _normalize_grayscale(bgr)
    roi_candidates = _generate_roi_candidates(grayscale)
    raw_roi_candidates = _generate_roi_candidates(raw_grayscale)
    image_candidates = _build_image_candidates(roi_candidates)

    hits: list[_DecodeHit] = []
    reconstructed_candidates, reconstructed_hits = _build_reconstructed_candidates(raw_roi_candidates)
    image_candidates = [*reconstructed_candidates, *image_candidates]
    hits.extend(reconstructed_hits)

    zxing_candidates = image_candidates[:90]
    pylibdmtx_candidates = image_candidates[:36]

    zxing_stage_hits: set[str] = set()
    for candidate in zxing_candidates:
        candidate_hits = _decode_with_zxing_candidate(candidate)
        if candidate_hits:
            zxing_stage_hits.add(candidate.stage_name)
            hits.extend(candidate_hits)

    supplemental = [candidate for candidate in zxing_candidates if candidate.stage_name in zxing_stage_hits]
    for candidate in _unique_candidates([*supplemental, *pylibdmtx_candidates]):
        hits.extend(_decode_with_pylibdmtx_candidate(candidate))

    if hits:
        primary, alternatives = _rank_hits(hits)
        processed_image = _find_stage_image(image_candidates, primary.stage_name)
        return DecodeResult(
            text=primary.text,
            engine=primary.engine,
            stage_name=primary.stage_name,
            score=primary.score,
            processed_image=processed_image,
            alternatives=alternatives,
        )

    fallback = max(image_candidates, key=lambda item: (item.preview_score, item.structural_score))
    return DecodeResult(
        text=None,
        engine=None,
        stage_name=fallback.stage_name,
        score=0.0,
        processed_image=fallback.image,
        alternatives=[],
    )


def _load_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.1)
    sharpened = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)
    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)


def _generate_roi_candidates(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    height, width = gray.shape
    rois: list[tuple[str, np.ndarray]] = [("full", gray)]
    proposed: list[tuple[float, str, np.ndarray]] = []
    proposed.extend(_generate_inset_rois(gray))
    proposed.extend(_generate_row_localized_rois(gray))

    for invert in (False, True):
        base = 255 - gray if invert else gray
        blurred = cv2.GaussianBlur(base, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
        closed = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if area < 0.08 * width * height or area > 0.98 * width * height:
                continue
            ratio = w / max(h, 1)
            if ratio < 0.65 or ratio > 1.35:
                continue

            pad = int(max(w, h) * 0.08)
            x0 = max(x - pad, 0)
            y0 = max(y - pad, 0)
            x1 = min(x + w + pad, width)
            y1 = min(y + h + pad, height)
            roi = gray[y0:y1, x0:x1]
            key = f"contour-{int(invert)}-{index}"
            proposed.append((area, key, roi))

            rect = cv2.minAreaRect(contour)
            deskewed = _extract_deskewed_roi(gray, rect)
            if deskewed is not None:
                proposed.append((area * 1.05, f"{key}-deskew", deskewed))

    seen_shapes: set[tuple[int, int, int]] = set()
    for _, name, roi in sorted(proposed, key=lambda item: item[0], reverse=True):
        if roi.size == 0:
            continue
        if name.startswith("row20-"):
            rois.append((name, roi))
            if len(rois) >= 10:
                break
            continue
        shape_key = (roi.shape[0] // 10, roi.shape[1] // 10, int(np.mean(roi)) // 8)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        rois.append((name, roi))
        if len(rois) >= 7:
            break

    return rois


def _generate_inset_rois(gray: np.ndarray) -> list[tuple[float, str, np.ndarray]]:
    height, width = gray.shape
    proposals: list[tuple[float, str, np.ndarray]] = []
    min_side = min(height, width)

    for inset_ratio in (0.03, 0.05, 0.08, 0.1, 0.12):
        inset_x = int(width * inset_ratio)
        inset_y = int(height * inset_ratio)
        x0 = inset_x
        y0 = inset_y
        x1 = max(width - inset_x, x0 + min_side // 2)
        y1 = max(height - inset_y, y0 + min_side // 2)
        if x1 - x0 < 120 or y1 - y0 < 120:
            continue
        crop = gray[y0:y1, x0:x1]
        proposals.append((float(crop.shape[0] * crop.shape[1]), f"inset-{inset_ratio:.2f}", crop))

    # Square-centered crops work well when the user already supplied a loose crop.
    for inset_ratio in (0.0, 0.04, 0.08, 0.12):
        side = int(min_side * (1.0 - inset_ratio))
        if side < 120:
            continue
        center_x = width / 2
        center_y = height / 2
        for shift_x_ratio in (-0.06, 0.0, 0.06):
            for shift_y_ratio in (-0.04, 0.0, 0.04):
                x0 = int(round(center_x - side / 2 + width * shift_x_ratio))
                y0 = int(round(center_y - side / 2 + height * shift_y_ratio))
                x0 = min(max(x0, 0), max(width - side, 0))
                y0 = min(max(y0, 0), max(height - side, 0))
                x1 = x0 + side
                y1 = y0 + side
                crop = gray[y0:y1, x0:x1]
                proposals.append(
                    (
                        float(side * side),
                        f"square-{inset_ratio:.2f}-sx{shift_x_ratio:+.2f}-sy{shift_y_ratio:+.2f}",
                        crop,
                    )
                )

    return proposals


def _generate_row_localized_rois(gray: np.ndarray) -> list[tuple[float, str, np.ndarray]]:
    module_count = 20
    points: list[tuple[float, float]] = []
    proposals: list[tuple[float, str, np.ndarray]] = []
    for variant_name, variant in _iter_row_localization_variants(gray):
        for response_name, response in _build_reconstruction_responses(variant):
            for quantile in (0.88, 0.9, 0.92):
                threshold_value = np.quantile(response, quantile)
                binary = (response >= threshold_value).astype(np.uint8)
                binary = cv2.morphologyEx(
                    binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                )
                component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

                points.clear()
                for index in range(1, component_count):
                    area = int(stats[index, cv2.CC_STAT_AREA])
                    if 3 <= area <= 120:
                        x, y = centroids[index]
                        points.append((float(x), float(y)))

                for row_index, row in enumerate(_cluster_rows(points, tolerance=6.0)):
                    centers_x = _dedupe_x([point[0] for point in row["pts"]])
                    if not (module_count - 2 <= len(centers_x) <= module_count + 2):
                        continue
                    pitch_samples = np.diff(centers_x)
                    pitch_samples = pitch_samples[(pitch_samples > 8.0) & (pitch_samples < 18.0)]
                    if len(pitch_samples) < 6:
                        continue
                    pitch = float(np.median(pitch_samples))
                    if not 10.0 <= pitch <= 16.0:
                        continue
                    fitted_centers = _fit_progression_subset(centers_x, module_count, pitch)
                    if fitted_centers is None:
                        continue

                    x0 = int(max(min(fitted_centers) - 2.0 * pitch, 0))
                    x1 = int(min(max(fitted_centers) + 1.8 * pitch, variant.shape[1]))
                    y0 = int(max(row["y_mean"] - 1.5 * pitch, 0))
                    y1 = int(min(row["y_mean"] + (module_count + 1.5) * pitch, variant.shape[0]))
                    if x1 - x0 < 140 or y1 - y0 < 140:
                        continue
                    crop = variant[y0:y1, x0:x1]
                    proposals.append(
                        (
                            float(crop.shape[0] * crop.shape[1]),
                            f"row20-{variant_name}-{response_name}-{quantile:.2f}-{row_index}",
                            crop,
                        )
                    )

        if proposals:
            break

    return proposals[:8]


def _iter_row_localization_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("base", gray)]
    for angle in (-10.0, -7.0, -4.0, 4.0, 7.0, 10.0):
        variants.append((f"rot{angle:+.0f}", _rotate_image_bound(gray, angle)))
    return variants


def _cluster_rows(points: Iterable[tuple[float, float]], tolerance: float = 4.5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for x, y in sorted(points, key=lambda point: point[1]):
        placed = False
        for row in rows:
            if abs(y - float(row["y_mean"])) <= tolerance:
                row_points = row["pts"]
                assert isinstance(row_points, list)
                row_points.append((x, y))
                row["y_mean"] = sum(point[1] for point in row_points) / len(row_points)
                placed = True
                break
        if not placed:
            rows.append({"y_mean": y, "pts": [(x, y)]})
    return rows


def _dedupe_x(values: Iterable[float], tolerance: float = 7.0) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _fit_progression_subset(
    centers_x: Iterable[float], module_count: int, base_pitch: float
) -> list[float] | None:
    values = sorted(centers_x)
    best_score = -1
    best_progression: list[float] | None = None

    for start in values:
        for pitch_delta in (-1.0, -0.5, 0.0, 0.5, 1.0):
            pitch = base_pitch + pitch_delta
            expected = [start + index * pitch for index in range(module_count)]
            matched = 0
            progression: list[float] = []
            for center in expected:
                nearest = min(values, key=lambda value: abs(value - center))
                if abs(nearest - center) <= 4.5:
                    matched += 1
                    progression.append(nearest)
                else:
                    progression.append(center)
            if matched > best_score:
                best_score = matched
                best_progression = progression

    if best_score < module_count - 3:
        return None
    return best_progression


def _rotate_image_bound(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    bound_w = int((height * sin) + (width * cos))
    bound_h = int((height * cos) + (width * sin))
    matrix[0, 2] += (bound_w / 2) - center[0]
    matrix[1, 2] += (bound_h / 2) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _build_reconstruction_responses(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(roi)
    sharpen = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpen, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)
    blackhat = cv2.morphologyEx(
        sharpen, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ).astype(np.float32)

    return [
        ("mix", cv2.normalize(0.65 * tophat + 0.35 * blackhat, None, 0, 1, cv2.NORM_MINMAX)),
        ("tophat", cv2.normalize(tophat, None, 0, 1, cv2.NORM_MINMAX)),
        ("blackhat", cv2.normalize(blackhat, None, 0, 1, cv2.NORM_MINMAX)),
    ]


def _extract_deskewed_roi(gray: np.ndarray, rect: tuple) -> np.ndarray | None:
    (center_x, center_y), (width, height), angle = rect
    if width < 20 or height < 20:
        return None
    if width < height:
        angle += 90.0
        width, height = height, width

    rotation = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        rotation,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    x0 = int(max(center_x - width / 2, 0))
    y0 = int(max(center_y - height / 2, 0))
    x1 = int(min(center_x + width / 2, gray.shape[1]))
    y1 = int(min(center_y + height / 2, gray.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return None
    return rotated[y0:y1, x0:x1]


def _build_image_candidates(roi_candidates: Iterable[tuple[str, np.ndarray]]) -> list[_ImageCandidate]:
    variants: list[_ImageCandidate] = []
    for roi_name, roi in roi_candidates:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(roi)
        blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
        sharpen = cv2.addWeighted(clahe, 1.9, blur, -0.9, 0)

        bases = {
            "gray": roi,
            "clahe": clahe,
            "sharp": sharpen,
            "inv": 255 - sharpen,
        }

        for base_name, base in bases.items():
            processed = {
                "base": base,
                "otsu": cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                "adapt": cv2.adaptiveThreshold(
                    base,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    3,
                ),
            }

            for proc_name, proc_img in processed.items():
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                morphs = {
                    "raw": proc_img,
                    "open": cv2.morphologyEx(proc_img, cv2.MORPH_OPEN, kernel),
                    "close": cv2.morphologyEx(proc_img, cv2.MORPH_CLOSE, kernel),
                }

                for morph_name, morph_img in morphs.items():
                    for scale in (2, 3, 4):
                        scaled = cv2.resize(
                            morph_img,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for rotation in (0, 90, 180, 270):
                            rotated = _rotate_orthogonal(scaled, rotation)
                            stage_name = (
                                f"{roi_name}:{base_name}:{proc_name}:{morph_name}:"
                                f"x{scale}:r{rotation}"
                            )
                            preview_score = _preview_score(rotated)
                            structural_score = _structural_score(rotated)
                            variants.append(
                                _ImageCandidate(
                                    stage_name=stage_name,
                                    image=rotated,
                                    preview_score=preview_score,
                                    structural_score=structural_score,
                                )
                            )

    variants.sort(key=lambda item: (item.structural_score, item.preview_score), reverse=True)
    return variants[:120]


def _build_reconstructed_candidates(
    roi_candidates: Iterable[tuple[str, np.ndarray]],
) -> tuple[list[_ImageCandidate], list[_DecodeHit]]:
    candidates: list[_ImageCandidate] = []
    hits: list[_DecodeHit] = []

    for roi_name, roi in list(roi_candidates)[:8]:
        reconstructed = _reconstruct_datamatrix_candidates(roi_name, roi)
        for stage_name, image, decoded_texts in reconstructed:
            preview_score = _preview_score(image)
            structural_score = _structural_score(image)
            candidates.append(
                _ImageCandidate(
                    stage_name=stage_name,
                    image=image,
                    preview_score=preview_score,
                    structural_score=structural_score,
                )
            )
            for text in decoded_texts:
                hits.append(
                    _DecodeHit(
                        text=text,
                        engine="zxingcpp-reconstruct",
                        stage_name=stage_name,
                        valid=True,
                        readable_ratio=_readable_ratio(text),
                        preview_score=preview_score,
                        structural_score=structural_score,
                    )
                )

    return candidates, hits


def _reconstruct_datamatrix_candidates(
    roi_name: str, roi: np.ndarray
) -> list[tuple[str, np.ndarray, list[str]]]:
    module_count = 20
    if min(roi.shape[:2]) < 180:
        return []

    reconstructed: list[tuple[str, np.ndarray, list[str]]] = []
    for response_name, response in _build_reconstruction_responses(roi):
        strip_height = max(16, min(28, roi.shape[0] // 12))
        top_projection = response[:strip_height, :].mean(axis=0)
        min_distance = max(8, roi.shape[1] // 36)

        for projection_quantile in (0.65, 0.7, 0.75, 0.8):
            threshold = float(np.quantile(top_projection, projection_quantile))
            top_centers = np.array(
                _find_peaks_1d(top_projection, min_dist=min_distance, threshold=threshold),
                dtype=np.float32,
            )
            if not (module_count - 2 <= len(top_centers) <= module_count + 2):
                continue

            pitch_samples = np.diff(top_centers)
            pitch_samples = pitch_samples[(pitch_samples > 8.0) & (pitch_samples < 18.0)]
            if len(pitch_samples) < 6:
                continue
            base_pitch = float(np.median(pitch_samples))
            if not 10.0 <= base_pitch <= 16.0:
                continue

            fitted_centers = _fit_progression_subset(top_centers.tolist(), module_count, base_pitch)
            if fitted_centers is None:
                continue
            top_centers = np.array(fitted_centers, dtype=np.float32)

            max_vertical_offset = max(12, int(base_pitch * 2.3))
            vertical_offsets = range(2, max_vertical_offset + 1, 2)

            for vertical_offset in vertical_offsets:
                for pitch in (
                    base_pitch - 0.4,
                    base_pitch - 0.2,
                    base_pitch,
                    base_pitch + 0.2,
                    base_pitch + 0.4,
                ):
                    for shear in (-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2):
                        scores = np.zeros((module_count, module_count), dtype=np.float32)
                        if not _fill_reconstruction_scores(
                            scores, response, top_centers, vertical_offset, pitch, shear
                        ):
                            continue

                        for quantile in (0.45, 0.5, 0.55, 0.6, 0.65):
                            threshold_value = float(np.quantile(scores, quantile))
                            bits = (scores >= threshold_value).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            _, oriented = _orient_bits(bits)
                            rendered = _render_bits(oriented)
                            decoded = _decode_pure_render(rendered)
                            if decoded:
                                stage_name = (
                                    f"{roi_name}:reconstruct:{response_name}:pq{projection_quantile:.2f}:"
                                    f"y{vertical_offset}:p{pitch:.2f}:s{shear:.2f}:q{quantile:.2f}"
                                )
                                reconstructed.append((stage_name, rendered, decoded))

    return reconstructed[:12]


def _fill_reconstruction_scores(
    scores: np.ndarray,
    response: np.ndarray,
    top_centers: np.ndarray,
    vertical_offset: float,
    pitch: float,
    shear: float,
) -> bool:
    for row in range(scores.shape[0]):
        center_y = int(round(vertical_offset + row * pitch))
        if center_y < 0 or center_y >= response.shape[0]:
            return False

        row_centers = top_centers + row * shear
        if row_centers.min() < 0 or row_centers.max() >= response.shape[1]:
            return False

        for col, center_x_float in enumerate(row_centers):
            center_x = int(round(float(center_x_float)))
            y0 = max(0, center_y - 3)
            y1 = min(response.shape[0], center_y + 4)
            x0 = max(0, center_x - 4)
            x1 = min(response.shape[1], center_x + 5)
            patch = response[y0:y1, x0:x1]
            scores[row, col] = float(patch.max()) if patch.size else 0.0

    return True


def _orient_bits(bits: np.ndarray) -> tuple[float, np.ndarray]:
    best: tuple[float, np.ndarray] | None = None
    for rotation in range(4):
        rotated = np.rot90(bits, rotation)
        left = rotated[:, 0]
        bottom = rotated[-1, :]
        top = rotated[0, :]
        right = rotated[:, -1]
        solid = (left.mean() + bottom.mean()) / 2
        alternating_top = np.mean(top[:-1] != top[1:]) if len(top) > 1 else 0.0
        alternating_right = np.mean(right[:-1] != right[1:]) if len(right) > 1 else 0.0
        score = solid * 2.9 + alternating_top * 1.25 + alternating_right * 1.25
        if best is None or score > best[0]:
            best = (score, rotated)

    assert best is not None
    return best


def _render_bits(bits: np.ndarray) -> np.ndarray:
    canvas = 255 * (1 - bits.astype(np.uint8))
    image = np.kron(canvas, np.ones((16, 16), dtype=np.uint8))
    return cv2.copyMakeBorder(image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)


def _decode_pure_render(rendered: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for candidate in (rendered, 255 - rendered):
        try:
            results = zxingcpp.read_barcodes(
                candidate,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=False,
                text_mode=zxingcpp.TextMode.Plain,
                is_pure=True,
            )
        except TypeError:
            results = zxingcpp.read_barcodes(candidate)

        for result in results or []:
            text = (getattr(result, "text", "") or "").strip()
            if text:
                decoded.append(text)

    return sorted(set(decoded))


def _find_peaks_1d(values: np.ndarray, min_dist: int, threshold: float) -> list[int]:
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        current = float(values[index])
        if current < threshold or current < values[index - 1] or current <= values[index + 1]:
            continue
        if peaks and index - peaks[-1] < min_dist:
            if current > float(values[peaks[-1]]):
                peaks[-1] = index
        else:
            peaks.append(index)
    return peaks


def _rotate_orthogonal(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _decode_with_zxing_candidate(candidate: _ImageCandidate) -> list[_DecodeHit]:
    hits: list[_DecodeHit] = []
    zxing_hits = _decode_with_zxing(candidate.image)
    for text, valid in zxing_hits:
        hits.append(
            _DecodeHit(
                text=text,
                engine="zxingcpp",
                stage_name=candidate.stage_name,
                valid=valid,
                readable_ratio=_readable_ratio(text),
                preview_score=candidate.preview_score,
                structural_score=candidate.structural_score,
            )
        )
    return hits


def _decode_with_pylibdmtx_candidate(candidate: _ImageCandidate) -> list[_DecodeHit]:
    hits: list[_DecodeHit] = []
    dmtx_hits = _decode_with_pylibdmtx(candidate.image)
    for text in dmtx_hits:
        hits.append(
            _DecodeHit(
                text=text,
                engine="pylibdmtx",
                stage_name=candidate.stage_name,
                valid=True,
                readable_ratio=_readable_ratio(text),
                preview_score=candidate.preview_score,
                structural_score=candidate.structural_score,
            )
        )
    return hits


def _unique_candidates(candidates: Iterable[_ImageCandidate]) -> list[_ImageCandidate]:
    seen: set[str] = set()
    unique: list[_ImageCandidate] = []
    for candidate in candidates:
        if candidate.stage_name in seen:
            continue
        seen.add(candidate.stage_name)
        unique.append(candidate)
    return unique


def _decode_with_zxing(image: np.ndarray) -> list[tuple[str, bool]]:
    try:
        results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=True,
            try_downscale=False,
            text_mode=zxingcpp.TextMode.Plain,
        )
    except TypeError:
        results = zxingcpp.read_barcodes(image)

    decoded: list[tuple[str, bool]] = []
    for result in results or []:
        if str(getattr(result, "format", "")).lower().replace(" ", "") not in {
            "datamatrix",
            "barcodeformat.datamatrix",
        }:
            continue
        text = (getattr(result, "text", "") or "").strip()
        if text:
            decoded.append((text, bool(getattr(result, "valid", True))))
    return decoded


def _decode_with_pylibdmtx(image: np.ndarray) -> list[str]:
    try:
        results = dmtx_decode(image, timeout=35, max_count=3, corrections=10)
    except Exception:
        return []

    decoded: list[str] = []
    for result in results or []:
        data = getattr(result, "data", b"")
        text = data.decode("utf-8", "replace").strip()
        if text:
            decoded.append(text)
    return decoded


def _rank_hits(hits: list[_DecodeHit]) -> tuple[CandidateResult, list[CandidateResult]]:
    grouped: dict[str, list[_DecodeHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.text, []).append(hit)

    ranked: list[tuple[float, str, list[_DecodeHit]]] = []
    for text, group in grouped.items():
        engines = {item.engine for item in group}
        base = max(_score_hit(item) for item in group)
        repeat_bonus = min(len(group), 5) * 0.45
        engine_bonus = 3.0 if len(engines) > 1 else 0.0
        ranked.append((base + repeat_bonus + engine_bonus, text, group))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_score, top_text, top_group = ranked[0]
    best_hit = max(top_group, key=_score_hit)
    primary = CandidateResult(
        text=top_text,
        engine=best_hit.engine,
        stage_name=best_hit.stage_name,
        score=round(top_score, 2),
    )

    alternatives: list[CandidateResult] = []
    for score, text, group in ranked[1:6]:
        best = max(group, key=_score_hit)
        alternatives.append(
            CandidateResult(
                text=text,
                engine=best.engine,
                stage_name=best.stage_name,
                score=round(score, 2),
            )
        )
    return primary, alternatives


def _score_hit(hit: _DecodeHit) -> float:
    return (
        5.0
        + (2.2 if hit.valid else 0.0)
        + hit.readable_ratio * 3.0
        + min(hit.preview_score / 90.0, 2.0)
        + min(hit.structural_score / 140.0, 2.0)
    )


def _find_stage_image(candidates: Iterable[_ImageCandidate], stage_name: str) -> np.ndarray:
    for candidate in candidates:
        if candidate.stage_name == stage_name:
            return candidate.image
    return next(iter(candidates)).image


def _preview_score(image: np.ndarray) -> float:
    stddev = float(np.std(image))
    laplacian = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return stddev + math.sqrt(max(laplacian, 0.0))


def _structural_score(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    density = float(np.count_nonzero(edges)) / max(edges.size, 1)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    fill_ratio = float(np.count_nonzero(binary)) / max(binary.size, 1)
    centered_fill = 1.0 - abs(fill_ratio - 0.5)
    return density * 180.0 + centered_fill * 55.0


def _readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable = sum(ch.isprintable() and not ch.isspace() for ch in text)
    return readable / len(text)
