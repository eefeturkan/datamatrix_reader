from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step03_blob_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionResult:
    name: str
    count: int
    matched: int
    precision: float
    recall: float
    f1: float
    points: np.ndarray


def extract_red_centers(marked_bgr: np.ndarray) -> np.ndarray:
    b = marked_bgr[:, :, 0].astype(np.int16)
    g = marked_bgr[:, :, 1].astype(np.int16)
    r = marked_bgr[:, :, 2].astype(np.int16)
    mask = (r > 160) & (r - g > 60) & (r - b > 60)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    points: list[tuple[float, float]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 400:
            x, y = centroids[idx]
            points.append((float(x), float(y)))
    return np.array(points, dtype=np.float32)


def build_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    gray_f = gray.astype(np.float32) / 255.0
    median31 = cv2.medianBlur(gray, 31).astype(np.float32) / 255.0
    median_dark_31 = np.clip(median31 - gray_f, 0.0, 1.0)
    radius = 5
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    sigma = 2.1
    ring = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ring /= float(ring.sum())
    matched_r5_pos = cv2.filter2D(median_dark_31, cv2.CV_32F, ring)
    matched_r5_pos = cv2.GaussianBlur(matched_r5_pos, (0, 0), 0.9)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    return {
        "gray_inv": 255 - gray,
        "median_dark_31": cv2.normalize(median_dark_31, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "matched_r5_pos": cv2.normalize(matched_r5_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "blackhat_17": blackhat,
    }


def match_metrics(pred: np.ndarray, truth: np.ndarray, tol: float = 6.0) -> tuple[int, float, float, float]:
    if len(pred) == 0:
        return 0, 0.0, 0.0, 0.0
    if len(truth) == 0:
        return 0, 0.0, 0.0, 0.0
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=2)
    used_truth: set[int] = set()
    matched = 0
    for i in range(d.shape[0]):
        j = int(np.argmin(d[i]))
        if d[i, j] <= tol and j not in used_truth:
            used_truth.add(j)
            matched += 1
    precision = matched / max(1, len(pred))
    recall = matched / max(1, len(truth))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return matched, precision, recall, f1


def draw_overlay(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, path: Path) -> None:
    vis = raw.copy()
    for x, y in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for x, y in pred:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def save_map(image: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), image)


def run_simple_blob(image: np.ndarray, *, min_area: float, max_area: float, blob_color: int, min_dist: float,
                    circularity: float | None = None, convexity: float | None = None,
                    inertia: float | None = None, min_threshold: float = 5, max_threshold: float = 220,
                    threshold_step: float = 10) -> np.ndarray:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = float(min_threshold)
    params.maxThreshold = float(max_threshold)
    params.thresholdStep = float(threshold_step)
    params.filterByColor = True
    params.blobColor = int(blob_color)
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.minDistBetweenBlobs = float(min_dist)
    params.filterByCircularity = circularity is not None
    if circularity is not None:
        params.minCircularity = float(circularity)
    params.filterByConvexity = convexity is not None
    if convexity is not None:
        params.minConvexity = float(convexity)
    params.filterByInertia = inertia is not None
    if inertia is not None:
        params.minInertiaRatio = float(inertia)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return pts


def run_hough(image: np.ndarray, *, dp: float, min_dist: float, p1: float, p2: float, min_radius: int, max_radius: int) -> np.ndarray:
    blur = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp,
        min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.round(circles[0, :, :2]).astype(np.float32)
    return arr


def evaluate(name: str, pred: np.ndarray, truth: np.ndarray, raw: np.ndarray) -> DetectionResult:
    matched, precision, recall, f1 = match_metrics(pred, truth)
    draw_overlay(raw, truth, pred, OUT_DIR / f"{name}_overlay.png")
    return DetectionResult(
        name=name,
        count=int(len(pred)),
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        points=pred,
    )


def main() -> None:
    raw = cv2.imread(str(ROOT / "cropped.png"))
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    if raw is None or marked is None:
        raise SystemExit("Missing input image.")

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    truth = extract_red_centers(marked)
    maps = build_maps(gray)
    for name, image in maps.items():
        save_map(image, OUT_DIR / f"{name}.png")

    experiments: list[tuple[str, np.ndarray]] = []
    experiments.append(("blob_gray_inv_loose", run_simple_blob(maps["gray_inv"], min_area=8, max_area=280, blob_color=255, min_dist=10)))
    experiments.append(("blob_gray_inv_circ", run_simple_blob(maps["gray_inv"], min_area=12, max_area=260, blob_color=255, min_dist=10, circularity=0.35)))
    experiments.append(("blob_median_dark_loose", run_simple_blob(maps["median_dark_31"], min_area=8, max_area=320, blob_color=255, min_dist=10)))
    experiments.append(("blob_median_dark_circ", run_simple_blob(maps["median_dark_31"], min_area=12, max_area=280, blob_color=255, min_dist=10, circularity=0.35)))
    experiments.append(("blob_matched_loose", run_simple_blob(maps["matched_r5_pos"], min_area=8, max_area=320, blob_color=255, min_dist=10)))
    experiments.append(("blob_matched_circ", run_simple_blob(maps["matched_r5_pos"], min_area=12, max_area=280, blob_color=255, min_dist=10, circularity=0.35)))
    experiments.append(("blob_blackhat_loose", run_simple_blob(maps["blackhat_17"], min_area=8, max_area=320, blob_color=255, min_dist=10)))
    experiments.append(("hough_gray_inv", run_hough(maps["gray_inv"], dp=1.0, min_dist=10, p1=80, p2=10, min_radius=2, max_radius=8)))
    experiments.append(("hough_median_dark", run_hough(maps["median_dark_31"], dp=1.0, min_dist=10, p1=80, p2=10, min_radius=2, max_radius=8)))
    experiments.append(("hough_matched", run_hough(maps["matched_r5_pos"], dp=1.0, min_dist=10, p1=80, p2=10, min_radius=2, max_radius=8)))

    results = [evaluate(name, pred, truth, raw) for name, pred in experiments]
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
