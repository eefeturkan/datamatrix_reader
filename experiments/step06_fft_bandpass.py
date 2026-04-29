from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step06_fft_bandpass"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalResult:
    name: str
    count: int
    matched: int
    precision: float
    recall: float
    f1: float


def extract_red_centers(marked_bgr: np.ndarray) -> np.ndarray:
    b = marked_bgr[:, :, 0].astype(np.int16)
    g = marked_bgr[:, :, 1].astype(np.int16)
    r = marked_bgr[:, :, 2].astype(np.int16)
    mask = (r > 160) & (r - g > 60) & (r - b > 60)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    pts: list[tuple[float, float]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 400:
            x, y = centroids[idx]
            pts.append((float(x), float(y)))
    return np.array(pts, dtype=np.float32)


def local_contrast_normalize(gray: np.ndarray, win: int = 41, eps: float = 1e-3) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(gray_f, (0, 0), win / 6.0)
    sq_mu = cv2.GaussianBlur(gray_f * gray_f, (0, 0), win / 6.0)
    sigma = np.sqrt(np.maximum(sq_mu - mu * mu, 0.0))
    return (gray_f - mu) / (sigma + eps)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fft_bandpass(image: np.ndarray, period: float, width: float, harmonic2: bool = False) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX * FX + FY * FY)
    f0 = 1.0 / period
    mask = np.exp(-0.5 * ((R - f0) / width) ** 2)
    if harmonic2:
        mask += 0.65 * np.exp(-0.5 * ((R - (2.0 * f0)) / (width * 1.2)) ** 2)
    mask = np.clip(mask, 0.0, 1.0)

    spectrum = np.fft.fft2(image)
    filtered = spectrum * mask
    recon = np.fft.ifft2(filtered)
    amp = np.abs(recon)
    return mask, amp


def topk_peaks(map_u8: np.ndarray, k: int, min_dist: float = 8.0) -> np.ndarray:
    f = map_u8.astype(np.float32)
    ys, xs = np.where(f > 0)
    scores = f[ys, xs]
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


def match_metrics(pred: np.ndarray, truth: np.ndarray, tol: float = 6.0) -> tuple[int, float, float, float]:
    if len(pred) == 0 or len(truth) == 0:
        return 0, 0.0, 0.0, 0.0
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=2)
    used: set[int] = set()
    matched = 0
    for i in range(d.shape[0]):
        j = int(np.argmin(d[i]))
        if d[i, j] <= tol and j not in used:
            used.add(j)
            matched += 1
    precision = matched / max(1, len(pred))
    recall = matched / max(1, len(truth))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    return matched, precision, recall, f1


def draw_overlay(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, path: Path) -> None:
    vis = raw.copy()
    for x, y in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for x, y in pred:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def evaluate(name: str, raw: np.ndarray, truth: np.ndarray, map_u8: np.ndarray, k: int = 220) -> EvalResult:
    pts = topk_peaks(map_u8, k=k, min_dist=8.0)
    matched, precision, recall, f1 = match_metrics(pts, truth)
    draw_overlay(raw, truth, pts, OUT_DIR / f"{name}_overlay.png")
    return EvalResult(name, len(pts), matched, precision, recall, f1)


def main() -> None:
    raw = cv2.imread(str(ROOT / "cropped.png"))
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    if raw is None or marked is None:
        raise SystemExit("Missing input images.")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    truth = extract_red_centers(marked)

    lcn = local_contrast_normalize(gray, win=41)
    dark = np.clip(-lcn, 0.0, None)
    dark_u8 = normalize_u8(dark)
    cv2.imwrite(str(OUT_DIR / "lcn_dark.png"), dark_u8)

    specs = [
        ("fft_p16p5_w0p010", 16.5, 0.010, False),
        ("fft_p17p0_w0p010", 17.0, 0.010, False),
        ("fft_p17p0_w0p014", 17.0, 0.014, False),
        ("fft_p17p0_h2", 17.0, 0.010, True),
        ("fft_p17p5_h2", 17.5, 0.010, True),
    ]

    results: list[EvalResult] = []
    for name, period, width, harmonic2 in specs:
        mask, amp = fft_bandpass(dark, period=period, width=width, harmonic2=harmonic2)
        mask_vis = normalize_u8(np.fft.fftshift(mask))
        amp_u8 = normalize_u8(amp)
        cv2.imwrite(str(OUT_DIR / f"{name}_mask.png"), mask_vis)
        cv2.imwrite(str(OUT_DIR / f"{name}_amp.png"), amp_u8)
        results.append(evaluate(name, raw, truth, amp_u8, k=220))

    results.sort(key=lambda x: x.f1, reverse=True)
    lines = [f"truth_count={len(truth)}"]
    for res in results:
        lines.append(
            f"{res.name}: count={res.count} matched={res.matched} "
            f"precision={res.precision:.3f} recall={res.recall:.3f} f1={res.f1:.3f}"
        )
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
