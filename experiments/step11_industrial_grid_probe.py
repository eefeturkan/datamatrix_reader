from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import zxingcpp

ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "industrial_grid_probe"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datamatrix_reader import pipeline as dm


@dataclass(frozen=True)
class GridProbe:
    image_name: str
    module_count: int
    u0: float
    v0: float
    quantile: float
    score: float
    occupancy: float
    decoded: tuple[str, ...]


def normalize_response(response: np.ndarray) -> np.ndarray:
    response = response.astype(np.float32)
    return (response - response.min()) / (response.max() - response.min() + 1e-6)


def combined_likelihood(feature_maps: dict[str, np.ndarray]) -> np.ndarray:
    channels = (
        ("hessian", 0.45),
        ("fft", 0.25),
        ("dark", 0.20),
        ("gray_inv", 0.10),
    )
    combined: np.ndarray | None = None
    for name, weight in channels:
        normalized = normalize_response(feature_maps[name])
        if combined is None:
            combined = weight * normalized
        else:
            combined += weight * normalized
    if combined is None:
        raise ValueError("No feature maps")
    return combined.astype(np.float32)


def decode_bits(bits: np.ndarray) -> tuple[str, ...]:
    _, oriented = dm._orient_bits(bits.astype(np.uint8))
    hits: list[str] = []
    for candidate in (oriented, 1 - oriented):
        rendered = dm._render_bits(candidate)
        hits.extend(result.text for result in zxingcpp.read_barcodes(rendered))
    return tuple(sorted(set(hits)))


def render_probe_bits(bits: np.ndarray, path: Path) -> None:
    _, oriented = dm._orient_bits(bits.astype(np.uint8))
    cv2.imwrite(str(path), dm._render_bits(oriented))


def draw_grid_overlay(
    image: np.ndarray,
    origin: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    module_count: int,
    u0: float,
    v0: float,
    path: Path,
) -> None:
    overlay = image.copy()
    corners = np.array(
        [
            origin + u0 * vx + v0 * vy,
            origin + (u0 + module_count - 1) * vx + v0 * vy,
            origin + (u0 + module_count - 1) * vx + (v0 + module_count - 1) * vy,
            origin + u0 * vx + (v0 + module_count - 1) * vy,
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
    for row in range(module_count):
        for col in range(module_count):
            point = origin + (u0 + col) * vx + (v0 + row) * vy
            cv2.circle(
                overlay,
                tuple(np.round(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), overlay)


def score_grid(bits: np.ndarray) -> tuple[float, np.ndarray]:
    orient_score, oriented = dm._orient_bits(bits.astype(np.uint8))
    finder_score = float(oriented[:, 0].mean() + oriented[-1, :].mean()) * 2.0
    timing_penalty = abs(float(oriented[0, :].mean()) - 0.5) + abs(
        float(oriented[:, -1].mean()) - 0.5
    )
    occupancy_penalty = abs(float(bits.mean()) - 0.5) * 1.2
    return float(orient_score + finder_score - timing_penalty - occupancy_penalty), oriented


def build_score_grid(
    origin: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    likelihood: np.ndarray,
    module_count: int,
    u0: float,
    v0: float,
) -> np.ndarray:
    rows, cols = np.mgrid[0:module_count, 0:module_count].astype(np.float32)
    xs = origin[0] + (u0 + cols) * vx[0] + (v0 + rows) * vy[0]
    ys = origin[1] + (u0 + cols) * vx[1] + (v0 + rows) * vy[1]
    return cv2.remap(
        likelihood,
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def probe_image(image_name: str) -> GridProbe:
    image_path = ROOT / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    weighted_points, weighted_scores, feature_maps = dm._build_bright_weighted_points(gray)
    best_fit = dm._fit_best_bright_frame(weighted_points, weighted_scores)
    if best_fit is None:
        raise RuntimeError(f"No bright frame fit for {image_name}")

    origin = np.asarray(best_fit.origin, dtype=np.float32)
    vx = np.asarray(best_fit.vx, dtype=np.float32)
    vy = np.asarray(best_fit.vy, dtype=np.float32)
    likelihood = combined_likelihood(feature_maps)

    best: tuple[float, int, float, float, float, np.ndarray, np.ndarray] | None = None
    for module_count in (20, 22, 24, 26):
        for u0 in np.arange(-4.0, 8.01, 0.75):
            for v0 in np.arange(-2.0, 8.01, 0.75):
                score_values = build_score_grid(
                    origin, vx, vy, likelihood, module_count, float(u0), float(v0)
                )
                for quantile in (0.42, 0.46, 0.50, 0.54):
                    threshold = float(np.quantile(score_values, quantile))
                    bits = (score_values >= threshold).astype(np.uint8)
                    structural_score, oriented = score_grid(bits)
                    current = (
                        structural_score,
                        module_count,
                        float(u0),
                        float(v0),
                        float(quantile),
                        bits,
                        oriented,
                    )
                    if best is None or current[0] > best[0]:
                        best = current

    if best is None:
        raise RuntimeError(f"No grid probe for {image_name}")

    score, module_count, u0, v0, quantile, bits, _ = best
    decoded = decode_bits(bits)

    stem = Path(image_name).stem
    render_probe_bits(bits, OUT_DIR / f"{stem}_best_bits.png")
    draw_grid_overlay(
        image,
        origin,
        vx,
        vy,
        module_count,
        u0,
        v0,
        OUT_DIR / f"{stem}_best_grid_overlay.png",
    )

    return GridProbe(
        image_name=image_name,
        module_count=module_count,
        u0=u0,
        v0=v0,
        quantile=quantile,
        score=score,
        occupancy=float(bits.mean()),
        decoded=decoded,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probes = [probe_image(name) for name in ("cropped.png", "yenitest.png")]
    lines = [
        "industrial_grid_probe",
        "method=bright dot likelihood + frame fit + module-count/offset search + Data Matrix border score",
        "",
    ]
    for probe in probes:
        lines.append(
            (
                f"{probe.image_name}: modules={probe.module_count} "
                f"u0={probe.u0:.2f} v0={probe.v0:.2f} q={probe.quantile:.2f} "
                f"score={probe.score:.3f} occupancy={probe.occupancy:.3f} "
                f"decoded={list(probe.decoded)}"
            )
        )
    summary = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
