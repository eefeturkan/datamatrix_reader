from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import zxingcpp


ROOT = Path(r"d:\DATAGUESS\datamatrix_v2")
OUT_DIR = ROOT / "artifacts" / "cropped_steps" / "step08_reconstruct_from_frame"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from datamatrix_reader.pipeline import _decode_pure_render, _orient_bits, _render_bits


def load_step07_module():
    path = ROOT / "experiments" / "step07_combined_edge_fit.py"
    spec = importlib.util.spec_from_file_location("step07_combined_edge_fit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_heatmap(scores: np.ndarray, path: Path) -> None:
    norm = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    big = cv2.resize(norm, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), big)


def draw_frame_overlay(base: np.ndarray, fit, path: Path) -> None:
    vis = base.copy()
    for row in range(20):
        for col in range(20):
            pt = fit.origin + col * fit.vx + row * fit.vy
            color = (0, 255, 0)
            if row in (0, 19) or col in (0, 19):
                color = (0, 255, 255)
            cv2.circle(vis, (int(round(float(pt[0]))), int(round(float(pt[1])))), 3, color, 1, lineType=cv2.LINE_AA)
    tl = fit.origin
    tr = fit.origin + 19 * fit.vx
    bl = fit.origin + 19 * fit.vy
    br = fit.origin + 19 * fit.vx + 19 * fit.vy
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [quad], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def sample_channel(channel: np.ndarray, fit, radius: int = 4, reducer: str = "max") -> np.ndarray:
    h, w = channel.shape[:2]
    scores = np.zeros((20, 20), dtype=np.float32)
    for row in range(20):
        for col in range(20):
            pt = fit.origin + col * fit.vx + row * fit.vy
            cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
            x0 = max(0, cx - radius)
            y0 = max(0, cy - radius)
            x1 = min(w, cx + radius + 1)
            y1 = min(h, cy + radius + 1)
            patch = channel[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if reducer == "mean":
                scores[row, col] = float(patch.mean())
            else:
                scores[row, col] = float(patch.max())
    return scores


def point_vote_scores(points: np.ndarray, weights: np.ndarray, fit) -> np.ndarray:
    basis = np.column_stack((fit.vx, fit.vy))
    inv = np.linalg.inv(basis)
    local = (points - fit.origin[None, :]) @ inv.T
    rounded = np.rint(local)
    err = np.linalg.norm(local - rounded, axis=1)
    good = err <= 0.34
    ij = rounded[good].astype(np.int32)
    w = weights[good] * (1.0 - (err[good] / 0.34))
    scores = np.zeros((20, 20), dtype=np.float32)
    for (i, j), ww in zip(ij, w):
        if 0 <= i < 20 and 0 <= j < 20:
            scores[j, i] += float(ww)
    return scores


def build_truth_matrix(marked_bgr: np.ndarray, fit) -> np.ndarray:
    mod = load_step07_module()
    truth = mod.extract_red_centers(marked_bgr)
    basis = np.column_stack((fit.vx, fit.vy))
    inv = np.linalg.inv(basis)
    local = (truth - fit.origin[None, :]) @ inv.T
    rounded = np.rint(local)
    err = np.linalg.norm(local - rounded, axis=1)
    good = err <= 0.36
    ij = rounded[good].astype(np.int32)
    bits = np.zeros((20, 20), dtype=np.uint8)
    for i, j in ij:
        if 0 <= i < 20 and 0 <= j < 20:
            bits[j, i] = 1
    return bits


def try_decodes(score: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, str | list[str]]]:
    results: list[tuple[str, np.ndarray, np.ndarray, str | list[str]]] = []
    for q in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56):
        thr = float(np.quantile(score, q))
        raw_bits = (score >= thr).astype(np.uint8)
        _, oriented = _orient_bits(raw_bits)
        rendered = _render_bits(oriented)
        decoded = [r.text for r in zxingcpp.read_barcodes(rendered, formats=zxingcpp.BarcodeFormat.DataMatrix)]
        results.append((f"q{int(round(q*100))}", raw_bits, oriented, decoded))
    return results


def decode_render(rendered: np.ndarray) -> list[str]:
    return [r.text for r in zxingcpp.read_barcodes(rendered, formats=zxingcpp.BarcodeFormat.DataMatrix)]


def uncertain_local_search(score: np.ndarray, bits: np.ndarray, quantile: float, max_depth: int = 3, top_n: int = 18) -> tuple[np.ndarray | None, list[str], str]:
    flat = bits.reshape(-1).copy()
    thr = float(np.quantile(score, quantile))
    uncertainty = np.abs(score.reshape(-1) - thr)
    idxs = np.argsort(uncertainty)[:top_n]

    base_render = _render_bits(bits)
    base_decoded = decode_render(base_render)
    if base_decoded:
        return bits.copy(), base_decoded, "base"

    import itertools

    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(idxs, depth):
            test = flat.copy()
            test[list(combo)] ^= 1
            test_bits = test.reshape(20, 20)
            rendered = _render_bits(test_bits)
            decoded = decode_render(rendered)
            if decoded:
                combo_cells = [(idx // 20, idx % 20) for idx in combo]
                return test_bits, decoded, f"depth={depth} flips={combo_cells}"
    return None, [], "not_found"


def main() -> None:
    mod = load_step07_module()
    raw = cv2.imread(str(ROOT / "cropped.png"))
    marked = cv2.imread(str(ROOT / "cropped_isaretli.png"))
    if raw is None or marked is None:
        raise SystemExit("Missing input images.")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    points, weights, images = mod.build_weighted_points(gray)
    best = None
    for vx, vy in mod.candidate_bases():
        try:
            fit = mod.fit_edges(points, weights, vx, vy)
        except RuntimeError:
            continue
        if best is None or fit.score > best.score:
            best = fit
    if best is None:
        raise SystemExit("No frame fit.")

    draw_frame_overlay(raw, best, OUT_DIR / "frame_overlay.png")

    lcn = mod.local_contrast_normalize(gray, win=41)
    dark = np.clip(-lcn, 0.0, None)
    dark_u8 = mod.normalize_u8(dark)
    hessian_u8 = mod.hessian_blob_multi(dark)
    fft_u8 = mod.fft_best_map(dark)
    gray_inv = 255 - gray

    vote = point_vote_scores(points, weights, best)
    hessian = sample_channel(hessian_u8.astype(np.float32) / 255.0, best, radius=4, reducer="max")
    fft = sample_channel(fft_u8.astype(np.float32) / 255.0, best, radius=4, reducer="max")
    dark_s = sample_channel(dark_u8.astype(np.float32) / 255.0, best, radius=3, reducer="mean")
    gray_inv_s = sample_channel(gray_inv.astype(np.float32) / 255.0, best, radius=3, reducer="mean")

    vote_n = vote / (vote.max() + 1e-6)
    score = 0.52 * vote_n + 0.20 * hessian + 0.14 * fft + 0.08 * dark_s + 0.06 * gray_inv_s

    save_heatmap(vote_n, OUT_DIR / "vote_heatmap.png")
    save_heatmap(hessian, OUT_DIR / "hessian_heatmap.png")
    save_heatmap(fft, OUT_DIR / "fft_heatmap.png")
    save_heatmap(score, OUT_DIR / "combined_score_heatmap.png")

    truth_bits = build_truth_matrix(marked, best)
    truth_render = _render_bits(truth_bits)
    truth_decode = _decode_pure_render(truth_render)
    cv2.imwrite(str(OUT_DIR / "truth_marked_matrix.png"), truth_render)

    results = try_decodes(score)
    lines = [
        f"origin=({best.origin[0]:.3f},{best.origin[1]:.3f})",
        f"vx=({best.vx[0]:.3f},{best.vx[1]:.3f})",
        f"vy=({best.vy[0]:.3f},{best.vy[1]:.3f})",
        f"truth_decode={decode_render(truth_render)!r}",
    ]
    cv2.imwrite(str(OUT_DIR / "truth_bits_render.png"), truth_render)

    best_search_bits = None
    best_search_text: list[str] = []
    best_search_note = ""
    for name, raw_bits, oriented_bits, decoded in results:
        rendered = _render_bits(oriented_bits)
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), rendered)
        lines.append(f"{name}={decoded!r}")
        if not decoded and name in {"q48", "q50"}:
            q_value = int(name[1:]) / 100.0
            fixed_bits, fixed_text, note = uncertain_local_search(score, raw_bits, q_value, max_depth=3, top_n=18)
            lines.append(f"{name}_local_search={fixed_text!r} {note}")
            if fixed_bits is not None and fixed_text:
                best_search_bits = fixed_bits
                best_search_text = fixed_text
                best_search_note = f"{name} {note}"
                cv2.imwrite(str(OUT_DIR / f"{name}_decoded_fix.png"), _render_bits(fixed_bits))

    if best_search_text:
        lines.append(f"final_decode={best_search_text!r}")
        lines.append(f"final_decode_note={best_search_note}")

    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
