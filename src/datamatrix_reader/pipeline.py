from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import itertools
import math
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode as dmtx_decode, encode as dmtx_encode
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


@dataclass(slots=True)
class _RenderedCandidate:
    stage_name: str
    image: np.ndarray
    bits: np.ndarray
    score: float


@dataclass(slots=True)
class _DecodedRenderedCandidate:
    stage_name: str
    image: np.ndarray
    bits: np.ndarray
    score: float
    decoded_texts: list[str]


@dataclass(frozen=True, slots=True)
class _ReconstructionProfile:
    response_names: frozenset[str]
    projection_quantiles: tuple[float, ...]
    pitch_deltas: tuple[float, ...]
    shear_values: tuple[float, ...]
    score_quantiles: tuple[float, ...]
    decode_limit: int
    max_decoded: int


_FAST_RECONSTRUCTION = _ReconstructionProfile(
    response_names=frozenset({"mix", "tophat"}),
    projection_quantiles=(0.65, 0.7),
    pitch_deltas=(-0.2, 0.0, 0.2),
    shear_values=(0.0, -0.8, -1.0),
    score_quantiles=(0.5,),
    decode_limit=6,
    max_decoded=12,
)

_REFINE_RECONSTRUCTION = _ReconstructionProfile(
    response_names=frozenset({"mix", "tophat"}),
    projection_quantiles=(0.65, 0.7),
    pitch_deltas=(-0.4, -0.2, 0.0, 0.2, 0.4),
    shear_values=(0.0, -0.8, -1.0, -0.6, -1.2),
    score_quantiles=(0.5, 0.55),
    decode_limit=18,
    max_decoded=18,
)

_FULL_RECONSTRUCTION = _ReconstructionProfile(
    response_names=frozenset({"mix", "tophat", "blackhat"}),
    projection_quantiles=(0.65, 0.7, 0.75, 0.8),
    pitch_deltas=(-0.4, -0.2, 0.0, 0.2, 0.4),
    shear_values=(-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2),
    score_quantiles=(0.45, 0.5, 0.55, 0.6, 0.65),
    decode_limit=12,
    max_decoded=12,
)


@dataclass(frozen=True, slots=True)
class _BrightGridCandidate:
    stage_name: str
    bits: np.ndarray
    score_grid: np.ndarray
    threshold_value: float
    score: float


@dataclass(frozen=True, slots=True)
class _BrightFrameFit:
    score: float
    origin: np.ndarray
    vx: np.ndarray
    vy: np.ndarray


def decode_image(image_bytes: bytes) -> DecodeResult:
    bgr = _load_image(image_bytes)
    raw_grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grayscale = _normalize_grayscale(bgr)
    bright_fallback_candidates: list[_ImageCandidate] = []

    for roi_candidates, fast_only in (
        (_generate_roi_candidates(raw_grayscale, fast_only=True), True),
        (_generate_roi_candidates(raw_grayscale, fast_only=False), False),
    ):
        reconstructed_candidates, reconstructed_hits = _build_reconstructed_candidates(
            roi_candidates, fast_only=fast_only
        )
        if reconstructed_hits:
            primary, alternatives = _rank_hits(reconstructed_hits)
            processed_image = _find_stage_image(reconstructed_candidates, primary.stage_name)
            return DecodeResult(
                text=primary.text,
                engine=primary.engine,
                stage_name=primary.stage_name,
                score=primary.score,
                processed_image=processed_image,
                alternatives=alternatives,
            )

    roi_candidates = _generate_roi_candidates(grayscale, fast_only=True)
    image_candidates = _build_image_candidates(roi_candidates)

    hits: list[_DecodeHit] = []

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

    if _looks_like_bright_dotpeen(raw_grayscale):
        bright_fallback_candidates, bright_hits = _build_bright_background_candidates(raw_grayscale)
        if bright_hits:
            primary, alternatives = _rank_hits(bright_hits)
            processed_image = _find_stage_image(bright_fallback_candidates, primary.stage_name)
            return DecodeResult(
                text=primary.text,
                engine=primary.engine,
                stage_name=primary.stage_name,
                score=primary.score,
                processed_image=processed_image,
                alternatives=alternatives,
            )

    fallback_pool = [*image_candidates, *bright_fallback_candidates]
    fallback = max(fallback_pool, key=lambda item: (item.preview_score, item.structural_score))
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


def _generate_roi_candidates(gray: np.ndarray, fast_only: bool = False) -> list[tuple[str, np.ndarray]]:
    height, width = gray.shape
    rois: list[tuple[str, np.ndarray]] = [("full", gray)]
    proposed: list[tuple[float, str, np.ndarray]] = []
    if fast_only:
        proposed.extend(_generate_inset_rois(gray, fast_only=True))
        proposed.extend(_generate_row_localized_rois(gray, fast_only=True))
    else:
        proposed.extend(_generate_inset_rois(gray, fast_only=False))
        proposed.extend(_generate_row_localized_rois(gray, fast_only=False))

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
            if len(rois) >= (6 if fast_only else 10):
                break
            continue
        shape_key = (roi.shape[0] // 10, roi.shape[1] // 10, int(np.mean(roi)) // 8)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        rois.append((name, roi))
        if len(rois) >= (5 if fast_only else 7):
            break

    return rois


def _generate_inset_rois(gray: np.ndarray, fast_only: bool = False) -> list[tuple[float, str, np.ndarray]]:
    height, width = gray.shape
    proposals: list[tuple[float, str, np.ndarray]] = []
    min_side = min(height, width)

    inset_ratios = (0.08,) if fast_only else (0.03, 0.05, 0.08, 0.1, 0.12)
    for inset_ratio in inset_ratios:
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
    square_insets = (0.08,) if fast_only else (0.0, 0.04, 0.08, 0.12)
    for inset_ratio in square_insets:
        side = int(min_side * (1.0 - inset_ratio))
        if side < 120:
            continue
        center_x = width / 2
        center_y = height / 2
        shift_x_values = (-0.06, 0.0) if fast_only else (-0.06, 0.0, 0.06)
        shift_y_values = (-0.04, 0.0) if fast_only else (-0.04, 0.0, 0.04)
        for shift_x_ratio in shift_x_values:
            for shift_y_ratio in shift_y_values:
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


def _generate_row_localized_rois(
    gray: np.ndarray, fast_only: bool = False
) -> list[tuple[float, str, np.ndarray]]:
    module_count = 20
    points: list[tuple[float, float]] = []
    proposals: list[tuple[float, str, np.ndarray]] = []
    for variant_name, variant in _iter_row_localization_variants(gray, fast_only=fast_only):
        responses = _build_reconstruction_responses(variant)
        if fast_only:
            responses = [item for item in responses if item[0] in {"mix", "tophat"}]
        quantiles = (0.9, 0.92) if fast_only else (0.88, 0.9, 0.92)
        for response_name, response in responses:
            for quantile in quantiles:
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


def _iter_row_localization_variants(
    gray: np.ndarray, fast_only: bool = False
) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("base", gray)]
    if fast_only:
        return variants
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
    fast_only: bool = False,
) -> tuple[list[_ImageCandidate], list[_DecodeHit]]:
    candidates: list[_ImageCandidate] = []
    hits: list[_DecodeHit] = []
    decoded_candidates: list[_DecodedRenderedCandidate] = []

    prioritized_rois = sorted(
        list(roi_candidates),
        key=lambda item: (
            0
            if item[0].startswith("row20-")
            else 1
            if item[0].startswith("inset-0.08")
            else 2
            if item[0].startswith("square-0.08")
            else 3,
            -(item[1].shape[0] * item[1].shape[1]),
        ),
    )
    roi_lookup = {name: roi for name, roi in prioritized_rois}
    profile = _FAST_RECONSTRUCTION if fast_only else _FULL_RECONSTRUCTION

    for roi_name, roi in prioritized_rois[: (2 if fast_only else 4)]:
        reconstructed = _reconstruct_datamatrix_candidates(roi_name, roi, profile=profile)
        decoded_candidates.extend(reconstructed)
        if decoded_candidates:
            break

    if fast_only and decoded_candidates:
        for roi_name in _ambiguous_reconstruction_rois(decoded_candidates):
            roi = roi_lookup.get(roi_name)
            if roi is None:
                continue
            decoded_candidates.extend(
                _reconstruct_datamatrix_candidates(
                    roi_name,
                    roi,
                    profile=_REFINE_RECONSTRUCTION,
                )
            )

    decoded_candidates = _unique_decoded_rendered_candidates(decoded_candidates)
    grouped: dict[str, list[_DecodedRenderedCandidate]] = {}
    for decoded_candidate in decoded_candidates:
        for text in decoded_candidate.decoded_texts:
            grouped.setdefault(text, []).append(decoded_candidate)

    for text, group in grouped.items():
        selected = _select_decoded_group_candidate(group, text)
        preview_score = selected.score * 10.0
        structural_score = selected.score * 10.0
        candidates.append(
            _ImageCandidate(
                stage_name=selected.stage_name,
                image=selected.image,
                preview_score=preview_score,
                structural_score=structural_score,
            )
        )
        for _ in group:
            hits.append(
                _DecodeHit(
                    text=text,
                    engine="zxingcpp-reconstruct",
                    stage_name=selected.stage_name,
                    valid=True,
                    readable_ratio=_readable_ratio(text),
                    preview_score=preview_score,
                    structural_score=structural_score,
                )
            )

    return candidates, hits


def _reconstruct_datamatrix_candidates(
    roi_name: str, roi: np.ndarray, profile: _ReconstructionProfile
) -> list[_DecodedRenderedCandidate]:
    module_count = 20
    if min(roi.shape[:2]) < 180:
        return []

    ranked_candidates: list[_RenderedCandidate] = []
    responses = _build_reconstruction_responses(roi)
    responses = [item for item in responses if item[0] in profile.response_names]
    for response_name, response in responses:
        strip_height = max(16, min(28, roi.shape[0] // 12))
        top_projection = response[:strip_height, :].mean(axis=0)
        min_distance = max(8, roi.shape[1] // 36)

        for projection_quantile in profile.projection_quantiles:
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
            vertical_offsets = _vertical_offsets_for_profile(roi_name, max_vertical_offset, profile)
            pitch_values = tuple(base_pitch + delta for delta in profile.pitch_deltas)

            for vertical_offset in vertical_offsets:
                for pitch in pitch_values:
                    for shear in profile.shear_values:
                        scores = np.zeros((module_count, module_count), dtype=np.float32)
                        if not _fill_reconstruction_scores(
                            scores, response, top_centers, vertical_offset, pitch, shear
                        ):
                            continue

                        for quantile in profile.score_quantiles:
                            threshold_value = float(np.quantile(scores, quantile))
                            bits = (scores >= threshold_value).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            orient_score, oriented = _orient_bits(bits)
                            occupancy = float(bits.mean())
                            score = orient_score - abs(occupancy - 0.5) * 1.2
                            stage_name = (
                                f"{roi_name}:reconstruct:{response_name}:pq{projection_quantile:.2f}:"
                                f"y{vertical_offset}:p{pitch:.2f}:s{shear:.2f}:q{quantile:.2f}"
                            )
                            ranked_candidates.append(
                                _RenderedCandidate(
                                    stage_name=stage_name,
                                    image=_render_bits(oriented),
                                    bits=oriented.copy(),
                                    score=score,
                                )
                            )

    decoded_candidates: list[_DecodedRenderedCandidate] = []
    for candidate in _select_top_rendered_candidates(ranked_candidates, limit=profile.decode_limit):
        decoded = _decode_pure_render(candidate.image)
        if decoded:
            decoded_candidates.append(
                _DecodedRenderedCandidate(
                    stage_name=candidate.stage_name,
                    image=candidate.image,
                    bits=candidate.bits,
                    score=candidate.score,
                    decoded_texts=decoded,
                )
            )
            if len(decoded_candidates) >= profile.max_decoded:
                break

    return decoded_candidates


def _vertical_offsets_for_profile(
    roi_name: str, max_vertical_offset: int, profile: _ReconstructionProfile
) -> range | tuple[int, ...]:
    if profile is _FAST_RECONSTRUCTION:
        if roi_name == "full":
            return range(2, min(max_vertical_offset, 12) + 1, 2)
        return (12, 16, 20, 24)
    if profile is _REFINE_RECONSTRUCTION:
        if roi_name == "full":
            return range(2, min(max_vertical_offset, 18) + 1, 2)
        start = max(8, min(max_vertical_offset, 12))
        stop = min(max_vertical_offset, 28)
        return tuple(range(start, stop + 1, 2))
    return range(2, max_vertical_offset + 1, 2)


def _ambiguous_reconstruction_rois(
    decoded_candidates: Iterable[_DecodedRenderedCandidate],
) -> list[str]:
    grouped: dict[str, list[_DecodedRenderedCandidate]] = {}
    for candidate in decoded_candidates:
        for text in candidate.decoded_texts:
            grouped.setdefault(text, []).append(candidate)

    roi_names: list[str] = []
    seen: set[str] = set()
    for text, group in grouped.items():
        unique_bits = {candidate.bits.tobytes() for candidate in group}
        should_refine = len(unique_bits) > 1

        reference_bits = _reference_bits_for_text(text)
        if reference_bits is not None:
            distances = [
                _bit_distance(candidate.bits, reference_bits)
                for candidate in group
                if candidate.bits.shape == reference_bits.shape
            ]
            if distances and min(distances) > 0:
                should_refine = True

        if not should_refine:
            continue

        for candidate in group:
            roi_name = candidate.stage_name.split(":reconstruct:", 1)[0]
            if roi_name in seen:
                continue
            seen.add(roi_name)
            roi_names.append(roi_name)

    return roi_names


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


def _select_top_rendered_candidates(
    candidates: Iterable[_RenderedCandidate], limit: int
) -> list[_RenderedCandidate]:
    unique: dict[str, _RenderedCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        image_key = candidate.image.tobytes()
        if image_key in unique:
            continue
        unique[image_key] = candidate
        if len(unique) >= limit:
            break
    return list(unique.values())


def _unique_decoded_rendered_candidates(
    candidates: Iterable[_DecodedRenderedCandidate],
) -> list[_DecodedRenderedCandidate]:
    unique: dict[tuple[str, bytes, tuple[str, ...]], _DecodedRenderedCandidate] = {}
    for candidate in candidates:
        key = (candidate.stage_name, candidate.bits.tobytes(), tuple(candidate.decoded_texts))
        unique[key] = candidate
    return list(unique.values())


def _select_decoded_group_candidate(
    candidates: list[_DecodedRenderedCandidate], text: str
) -> _DecodedRenderedCandidate:
    reference_bits = _reference_bits_for_text(text)
    if reference_bits is None:
        return max(candidates, key=lambda item: item.score)

    comparable = [candidate for candidate in candidates if candidate.bits.shape == reference_bits.shape]
    if not comparable:
        return max(candidates, key=lambda item: item.score)

    return min(
        comparable,
        key=lambda item: (
            _bit_distance(item.bits, reference_bits),
            -item.score,
            item.stage_name,
        ),
    )


@lru_cache(maxsize=64)
def _reference_bits_for_text(text: str) -> np.ndarray | None:
    try:
        encoded = dmtx_encode(text.encode("utf-8"))
    except Exception:
        return None

    width = int(getattr(encoded, "width", 0))
    height = int(getattr(encoded, "height", 0))
    pixels = getattr(encoded, "pixels", b"")
    if width <= 0 or height <= 0 or not pixels:
        return None

    raster = np.frombuffer(pixels, dtype=np.uint8)
    channels = max(raster.size // max(width * height, 1), 1)
    try:
        raster = raster.reshape(height, width, channels)
    except ValueError:
        return None

    gray = raster[:, :, 0]
    mask = (gray < 128).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    module_count = 20
    if crop.shape[0] % module_count != 0 or crop.shape[1] % module_count != 0:
        return None

    module_height = crop.shape[0] // module_count
    module_width = crop.shape[1] // module_count
    if module_height <= 0 or module_width <= 0:
        return None

    bits = (
        crop.reshape(module_count, module_height, module_count, module_width).mean(axis=(1, 3))
        >= 0.5
    ).astype(np.uint8)
    _, oriented = _orient_bits(bits)
    return oriented


def _bit_distance(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape:
        return max(left.size, right.size)
    return int(np.count_nonzero(left != right))


def _looks_like_bright_dotpeen(gray: np.ndarray) -> bool:
    return float(np.mean(gray)) >= 110.0 and float(np.std(gray)) >= 20.0


def _build_bright_background_candidates(
    gray: np.ndarray,
) -> tuple[list[_ImageCandidate], list[_DecodeHit]]:
    bright_candidates = _search_bright_frame_candidates(gray)
    if not bright_candidates:
        return [], []

    image_candidates: list[_ImageCandidate] = []
    hits: list[_DecodeHit] = []

    for candidate in bright_candidates[:6]:
        rendered = _render_bits(candidate.bits)
        preview_score = candidate.score * 10.0
        structural_score = candidate.score * 10.0
        image_candidates.append(
            _ImageCandidate(
                stage_name=candidate.stage_name,
                image=rendered,
                preview_score=preview_score,
                structural_score=structural_score,
            )
        )

        decoded_texts = _decode_direct_render(rendered)
        if not decoded_texts:
            decoded_texts = _decode_bright_uncertain_cells(candidate)

        for text in decoded_texts:
            hits.append(
                _DecodeHit(
                    text=text,
                    engine="zxingcpp-bright-reconstruct",
                    stage_name=candidate.stage_name,
                    valid=True,
                    readable_ratio=_readable_ratio(text),
                    preview_score=preview_score,
                    structural_score=structural_score,
                )
            )

    return image_candidates, hits


def _search_bright_frame_candidates(gray: np.ndarray) -> list[_BrightGridCandidate]:
    weighted_points, weighted_scores, feature_maps = _build_bright_weighted_points(gray)
    if weighted_points.size == 0:
        return []

    best_fit = _fit_best_bright_frame(weighted_points, weighted_scores)
    if best_fit is None:
        return []

    vote = _bright_point_vote_scores(weighted_points, weighted_scores, best_fit)
    vote_norm = vote / (float(vote.max()) + 1e-6)
    hessian_scores = _sample_bright_channel(feature_maps["hessian"], best_fit, radius=4, reducer="max")
    fft_scores = _sample_bright_channel(feature_maps["fft"], best_fit, radius=4, reducer="max")
    dark_scores = _sample_bright_channel(feature_maps["dark"], best_fit, radius=3, reducer="mean")
    gray_inv_scores = _sample_bright_channel(feature_maps["gray_inv"], best_fit, radius=3, reducer="mean")
    combined = (
        0.52 * vote_norm
        + 0.20 * hessian_scores
        + 0.14 * fft_scores
        + 0.08 * dark_scores
        + 0.06 * gray_inv_scores
    ).astype(np.float32)

    candidates: list[_BrightGridCandidate] = []
    for quantile in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56):
        threshold_value = float(np.quantile(combined, quantile))
        bits = (combined >= threshold_value).astype(np.uint8)
        candidates.append(
            _BrightGridCandidate(
                stage_name=(
                    f"bright-frame:x{best_fit.origin[0]:.1f}:y{best_fit.origin[1]:.1f}:"
                    f"vx{best_fit.vx[0]:.2f},{best_fit.vx[1]:.2f}:"
                    f"vy{best_fit.vy[0]:.2f},{best_fit.vy[1]:.2f}:q{quantile:.2f}"
                ),
                bits=bits,
                score_grid=combined,
                threshold_value=threshold_value,
                score=best_fit.score,
            )
        )
    return candidates


def _search_bright_grid_candidates(gray: np.ndarray) -> list[_BrightGridCandidate]:
    responses = _build_bright_dot_responses(gray)
    if not responses:
        return []

    module_count = 20
    height, width = gray.shape
    seed_x0 = width * (63.0 / 464.0)
    seed_y0 = height * (18.0 / 423.0)
    seed_x_pitch = width * (((388.0 - 63.0) / 19.0) / 464.0)
    seed_y_pitch = height * (((366.0 - 18.0) / 19.0) / 423.0)

    candidates: list[_BrightGridCandidate] = []
    seen: set[bytes] = set()

    x0_values = np.arange(seed_x0 - 4.0, seed_x0 + 4.1, 1.0)
    y0_values = np.arange(seed_y0 - 4.0, seed_y0 + 4.1, 1.0)
    x_pitch_values = np.arange(seed_x_pitch - 0.9, seed_x_pitch + 0.91, 0.3)
    y_pitch_values = np.arange(seed_y_pitch - 0.9, seed_y_pitch + 0.91, 0.3)
    quantiles = (0.25, 0.35, 0.45, 0.55)

    max_per_response = 12
    for response_name, response in responses[:4]:
        local_count = 0
        for x0 in x0_values:
            for y0 in y0_values:
                for x_pitch in x_pitch_values:
                    for y_pitch in y_pitch_values:
                        score_grid = np.zeros((module_count, module_count), dtype=np.float32)
                        if not _fill_bright_score_grid(score_grid, response, x0, y0, x_pitch, y_pitch):
                            continue

                        top_mean = float(np.mean(score_grid[0, :]))
                        left_mean = float(np.mean(score_grid[:, 0]))
                        bottom_mean = float(np.mean(score_grid[-1, :]))
                        for quantile in quantiles:
                            threshold_value = float(np.quantile(score_grid, quantile))
                            bits = (score_grid >= threshold_value).astype(np.uint8)
                            bits[0, :] = 1
                            bits[:, 0] = 1
                            orient_score, oriented = _orient_bits(bits)
                            occupancy = float(oriented.mean())
                            score = (
                                orient_score
                                + top_mean
                                + left_mean
                                + bottom_mean
                                - abs(occupancy - 0.5) * 1.5
                            )
                            key = oriented.tobytes()
                            if key in seen:
                                continue
                            seen.add(key)
                            candidates.append(
                                _BrightGridCandidate(
                                    stage_name=(
                                        f"bright:{response_name}:x0{x0:.1f}:y0{y0:.1f}:"
                                        f"xp{x_pitch:.2f}:yp{y_pitch:.2f}:q{quantile:.2f}"
                                    ),
                                    bits=oriented.copy(),
                                    score_grid=score_grid.copy(),
                                    threshold_value=threshold_value,
                                    score=score,
                                )
                            )
                            local_count += 1
                            if local_count >= max_per_response:
                                break
                        if local_count >= max_per_response:
                            break
                    if local_count >= max_per_response:
                        break
                if local_count >= max_per_response:
                    break
            if local_count >= max_per_response:
                break

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:24]


def _build_bright_weighted_points(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    dark_lcn = _bright_dark_lcn(gray)
    dark_u8 = cv2.normalize(dark_lcn, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    gray_inv = ((255 - gray).astype(np.float32) / 255.0).astype(np.float32)
    hough = _bright_hough_centers((255 - gray).astype(np.uint8))
    hessian = _bright_topk_points(_bright_hessian_blob_multi(dark_lcn), 220)
    fft = _bright_topk_points(_bright_fft_best_map(dark_lcn), 220)

    points: list[np.ndarray] = []
    weights: list[float] = []
    for point in hough:
        points.append(point)
        weights.append(1.0)
    for point in hessian:
        points.append(point)
        weights.append(0.85)
    for point in fft:
        points.append(point)
        weights.append(0.80)

    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), {}

    merged_points, merged_weights = _merge_bright_points(
        np.array(points, dtype=np.float32),
        np.array(weights, dtype=np.float32),
        radius=5.5,
    )
    return merged_points, merged_weights, {
        "dark": dark_u8,
        "gray_inv": gray_inv,
        "hessian": _bright_hessian_blob_multi(dark_lcn).astype(np.float32) / 255.0,
        "fft": _bright_fft_best_map(dark_lcn).astype(np.float32) / 255.0,
    }


def _bright_dark_lcn(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray_f, (0, 0), 41 / 6.0)
    mean_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), 41 / 6.0)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    lcn = (gray_f - mean) / (std + 1e-3)
    return np.clip(-lcn, 0.0, None).astype(np.float32)


def _bright_hessian_blob_multi(dark_lcn: np.ndarray) -> np.ndarray:
    maps: list[np.ndarray] = []
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
        isotropy = np.minimum(np.abs(l1), np.abs(l2)) / (np.maximum(np.abs(l1), np.abs(l2)) + 1e-6)
        blobness = np.where(same_sign, np.sqrt(np.maximum(det, 0.0)) * isotropy, 0.0)
        maps.append(cv2.normalize(blobness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return np.max(np.stack(maps, axis=0), axis=0).astype(np.uint8)


def _bright_fft_best_map(dark_lcn: np.ndarray) -> np.ndarray:
    height, width = dark_lcn.shape
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftfreq(width)
    grid_x, grid_y = np.meshgrid(freq_x, freq_y)
    radial = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    center_freq = 1.0 / 17.0
    width_sigma = 0.014
    mask = np.exp(-0.5 * ((radial - center_freq) / width_sigma) ** 2)
    spectrum = np.fft.fft2(dark_lcn)
    recon = np.fft.ifft2(spectrum * mask)
    amplitude = np.abs(recon)
    return cv2.normalize(amplitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _bright_hough_centers(gray_inv: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(gray_inv, 5)
    circles = cv2.HoughCircles(
        blurred,
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


def _bright_topk_points(map_u8: np.ndarray, count: int, min_dist: float = 8.0) -> np.ndarray:
    rows, cols = np.where(map_u8 > 0)
    scores = map_u8[rows, cols].astype(np.float32)
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float]] = []
    min_dist_sq = min_dist * min_dist
    for index in order:
        x = float(cols[index])
        y = float(rows[index])
        if all((x - px) ** 2 + (y - py) ** 2 > min_dist_sq for px, py in selected):
            selected.append((x, y))
            if len(selected) >= count:
                break
    return np.array(selected, dtype=np.float32)


def _merge_bright_points(
    points: np.ndarray,
    weights: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(weights)[::-1]
    kept_points: list[np.ndarray] = []
    kept_weights: list[float] = []
    radius_sq = radius * radius
    for index in order:
        point = points[index]
        weight = float(weights[index])
        merged = False
        for candidate_index, kept_point in enumerate(kept_points):
            if float(np.sum((point - kept_point) ** 2)) <= radius_sq:
                total = kept_weights[candidate_index] + weight
                kept_points[candidate_index] = (kept_point * kept_weights[candidate_index] + point * weight) / total
                kept_weights[candidate_index] = total
                merged = True
                break
        if not merged:
            kept_points.append(point.copy())
            kept_weights.append(weight)
    return np.array(kept_points, dtype=np.float32), np.array(kept_weights, dtype=np.float32)


def _bright_candidate_bases() -> list[tuple[np.ndarray, np.ndarray]]:
    bases: list[tuple[np.ndarray, np.ndarray]] = []
    for angle_x in (3.0, 3.8, 4.6):
        for angle_y in (95.0, 96.5, 98.0):
            for scale_x in (15.8, 16.2):
                for scale_y in (16.0, 16.6):
                    vx = np.array(
                        [math.cos(math.radians(angle_x)) * scale_x, math.sin(math.radians(angle_x)) * scale_x],
                        dtype=np.float32,
                    )
                    vy = np.array(
                        [math.cos(math.radians(angle_y)) * scale_y, math.sin(math.radians(angle_y)) * scale_y],
                        dtype=np.float32,
                    )
                    bases.append((vx, vy))
    return bases


def _fit_best_bright_frame(points: np.ndarray, weights: np.ndarray) -> _BrightFrameFit | None:
    best: _BrightFrameFit | None = None
    for vx, vy in _bright_candidate_bases():
        fit = _fit_bright_edges(points, weights, vx, vy)
        if fit is not None and (best is None or fit.score > best.score):
            best = fit
    return best


def _fit_bright_edges(
    points: np.ndarray,
    weights: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> _BrightFrameFit | None:
    basis = np.column_stack((vx, vy))
    inv = np.linalg.inv(basis)
    coords = points @ inv.T
    best: _BrightFrameFit | None = None

    for offset_u in np.linspace(0.0, 0.92, 12):
        for offset_v in np.linspace(0.0, 0.92, 12):
            shifted = coords - np.array([offset_u, offset_v], dtype=np.float32)
            rounded = np.rint(shifted)
            error = np.linalg.norm(shifted - rounded, axis=1)
            good = error <= 0.34
            if int(np.sum(good)) < 120:
                continue

            ij = rounded[good].astype(np.int32)
            fit_weights = weights[good]
            min_i, max_i = int(ij[:, 0].min()), int(ij[:, 0].max())
            min_j, max_j = int(ij[:, 1].min()), int(ij[:, 1].max())
            if max_i - min_i < 19 or max_j - min_j < 19:
                continue

            i_scores: list[tuple[float, int]] = []
            for left in range(min_i, max_i - 18):
                right = left + 19
                i_scores.append(
                    (
                        float(fit_weights[ij[:, 0] == left].sum()) + float(fit_weights[ij[:, 0] == right].sum()),
                        left,
                    )
                )
            j_scores: list[tuple[float, int]] = []
            for top in range(min_j, max_j - 18):
                bottom = top + 19
                j_scores.append(
                    (
                        float(fit_weights[ij[:, 1] == top].sum()) + float(fit_weights[ij[:, 1] == bottom].sum()),
                        top,
                    )
                )
            top_lefts = [left for _, left in sorted(i_scores, reverse=True)[:5]]
            top_tops = [top for _, top in sorted(j_scores, reverse=True)[:5]]

            for left in top_lefts:
                right = left + 19
                within_i = (ij[:, 0] >= left) & (ij[:, 0] <= right)
                if np.sum(within_i) < 70:
                    continue
                for top in top_tops:
                    bottom = top + 19
                    in_box = within_i & (ij[:, 1] >= top) & (ij[:, 1] <= bottom)
                    if np.sum(in_box) < 90:
                        continue
                    box_ij = ij[in_box]
                    box_weights = fit_weights[in_box]
                    edge_score = (
                        float(box_weights[box_ij[:, 0] == left].sum())
                        + float(box_weights[box_ij[:, 0] == right].sum())
                        + float(box_weights[box_ij[:, 1] == top].sum())
                        + float(box_weights[box_ij[:, 1] == bottom].sum())
                    )
                    inside_score = float(box_weights.sum())
                    outside_score = float(fit_weights[~in_box].sum())
                    score = 8.0 * edge_score + 0.8 * inside_score - 0.85 * outside_score
                    origin = np.array([left + offset_u, top + offset_v], dtype=np.float32) @ basis.T
                    fit = _BrightFrameFit(
                        score=score,
                        origin=origin.astype(np.float32),
                        vx=vx,
                        vy=vy,
                    )
                    if best is None or fit.score > best.score:
                        best = fit
    return best


def _bright_point_vote_scores(points: np.ndarray, weights: np.ndarray, fit: _BrightFrameFit) -> np.ndarray:
    basis = np.column_stack((fit.vx, fit.vy))
    inv = np.linalg.inv(basis)
    local = (points - fit.origin[None, :]) @ inv.T
    rounded = np.rint(local)
    error = np.linalg.norm(local - rounded, axis=1)
    good = error <= 0.34
    ij = rounded[good].astype(np.int32)
    effective_weights = weights[good] * (1.0 - error[good] / 0.34)
    score = np.zeros((20, 20), dtype=np.float32)
    for (col, row), weight in zip(ij, effective_weights):
        if 0 <= row < 20 and 0 <= col < 20:
            score[row, col] += float(weight)
    return score


def _sample_bright_channel(
    channel: np.ndarray,
    fit: _BrightFrameFit,
    radius: int,
    reducer: str,
) -> np.ndarray:
    height, width = channel.shape[:2]
    score = np.zeros((20, 20), dtype=np.float32)
    for row in range(20):
        for col in range(20):
            point = fit.origin + col * fit.vx + row * fit.vy
            center_x = int(round(float(point[0])))
            center_y = int(round(float(point[1])))
            x0 = max(0, center_x - radius)
            y0 = max(0, center_y - radius)
            x1 = min(width, center_x + radius + 1)
            y1 = min(height, center_y + radius + 1)
            patch = channel[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            score[row, col] = float(patch.mean()) if reducer == "mean" else float(patch.max())
    return score


def _build_bright_dot_responses(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    responses: list[tuple[str, np.ndarray]] = []

    for kernel_size in (21, 31):
        background = cv2.medianBlur(smoothed, kernel_size)
        dark = cv2.subtract(background, smoothed)
        dark_norm = cv2.normalize(dark, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        responses.append((f"median_dark_{kernel_size}", dark_norm))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(smoothed)
    sharpened = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.8, 0)
    tophat = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(np.float32)
    blackhat = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(np.float32)
    responses.append(("smooth_blackhat", cv2.normalize(blackhat, None, 0, 1, cv2.NORM_MINMAX)))
    responses.append(("smooth_mix", cv2.normalize(0.65 * tophat + 0.35 * blackhat, None, 0, 1, cv2.NORM_MINMAX)))

    for radius in (4, 5, 6, 7):
        outer = np.zeros((2 * radius + 3, 2 * radius + 3), dtype=np.float32)
        inner = np.zeros_like(outer)
        cv2.circle(outer, (radius + 1, radius + 1), radius + 1, 1.0, -1)
        cv2.circle(inner, (radius + 1, radius + 1), max(1, radius - 1), 1.0, -1)
        ring = outer - inner
        inner_sum = float(inner.sum())
        ring_sum = float(ring.sum())
        if inner_sum <= 0 or ring_sum <= 0:
            continue
        matched = cv2.filter2D(smoothed.astype(np.float32), cv2.CV_32F, inner / inner_sum) - cv2.filter2D(
            smoothed.astype(np.float32), cv2.CV_32F, ring / ring_sum
        )
        signal = np.clip(matched, 0, None)
        normalized = cv2.normalize(signal, None, 0, 1, cv2.NORM_MINMAX)
        if normalized is not None and float(np.std(normalized)) > 0.05:
            responses.append((f"matched_r{radius}_pos", normalized.astype(np.float32)))

    return responses


def _fill_bright_score_grid(
    score_grid: np.ndarray,
    response: np.ndarray,
    x0: float,
    y0: float,
    x_pitch: float,
    y_pitch: float,
) -> bool:
    for row in range(score_grid.shape[0]):
        center_y = int(round(y0 + row * y_pitch))
        if center_y < 0 or center_y >= response.shape[0]:
            return False
        for col in range(score_grid.shape[1]):
            center_x = int(round(x0 + col * x_pitch))
            if center_x < 0 or center_x >= response.shape[1]:
                return False
            patch = response[
                max(0, center_y - 4) : min(response.shape[0], center_y + 5),
                max(0, center_x - 4) : min(response.shape[1], center_x + 5),
            ]
            score_grid[row, col] = float(patch.max()) if patch.size else 0.0
    return True


def _decode_bright_uncertain_cells(candidate: _BrightGridCandidate) -> list[str]:
    decoded: list[str] = []
    margins = np.abs(candidate.score_grid - candidate.threshold_value)
    uncertain_cells: list[tuple[float, int, int]] = []
    for row in range(candidate.score_grid.shape[0]):
        for col in range(candidate.score_grid.shape[1]):
            uncertain_cells.append((float(margins[row, col]), row, col))
    uncertain_cells.sort(key=lambda item: item[0])
    top_cells = [(row, col) for _, row, col in uncertain_cells[:18]]

    rendered = _render_bits(candidate.bits)
    decoded = _decode_direct_render(rendered)
    if decoded:
        return decoded

    for depth in (1, 2, 3):
        limit = 40_000
        for count, combo in enumerate(itertools.combinations(range(len(top_cells)), depth), start=1):
            trial = candidate.bits.copy()
            for index in combo:
                row, col = top_cells[index]
                trial[row, col] = 1 - trial[row, col]
            decoded.extend(_decode_direct_render(_render_bits(trial)))
            if decoded:
                return sorted(set(decoded))
            if count >= limit:
                break

    return []


def _decode_direct_render(rendered: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for candidate in (rendered, 255 - rendered):
        try:
            results = zxingcpp.read_barcodes(
                candidate,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=False,
                text_mode=zxingcpp.TextMode.Plain,
            )
        except TypeError:
            results = zxingcpp.read_barcodes(candidate)

        for result in results or []:
            text = (getattr(result, "text", "") or "").strip()
            if text:
                decoded.append(text)
    return sorted(set(decoded))


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
