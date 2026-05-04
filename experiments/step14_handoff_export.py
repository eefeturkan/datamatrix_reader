from __future__ import annotations

from pathlib import Path
import re
import shutil

from step13_od_then_range_grid_decode import OUT_DIR as OD_OUT_DIR
from step13_od_then_range_grid_decode import ROOT, run


EXPORT_DIR = ROOT / "artifacts" / "barcode_exports"


def safe_barcode_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or "decoded_barcode"


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def export_decoded_artifacts(
    image_name: str = "saf.png",
    model_name: str = "best.pt",
) -> Path:
    image_path = ROOT / image_name
    model_path = ROOT / model_name
    attempts = run(image_path=image_path, model_path=model_path)
    decoded_attempt = next((attempt for attempt in attempts if attempt.decoded), None)
    if decoded_attempt is None:
        raise RuntimeError("No decoded Data Matrix found")

    decoded_text = decoded_attempt.decoded[0]
    safe_name = safe_barcode_name(decoded_text)
    stem = image_path.stem
    det_index = decoded_attempt.detection.index
    quantile = decoded_attempt.quantile

    source_crop = OD_OUT_DIR / f"{stem}_det{det_index}_crop.png"
    source_pipeline = OD_OUT_DIR / f"{stem}_det{det_index}_pipeline.png"
    source_response = OD_OUT_DIR / f"{stem}_det{det_index}_range_response.png"
    source_bits = (
        OD_OUT_DIR / f"{stem}_det{det_index}_q{quantile:.2f}_bits.png"
        if quantile is not None
        else None
    )
    source_summary = OD_OUT_DIR / f"{stem}_summary.txt"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    copy_if_exists(source_crop, EXPORT_DIR / f"{safe_name}_crop.png")
    copy_if_exists(source_pipeline, EXPORT_DIR / f"{safe_name}_pipeline.png")
    copy_if_exists(source_response, EXPORT_DIR / f"{safe_name}_range_response.png")
    if source_bits is not None:
        copy_if_exists(source_bits, EXPORT_DIR / f"{safe_name}_bits.png")
    copy_if_exists(source_summary, EXPORT_DIR / f"{safe_name}_summary.txt")

    metadata_path = EXPORT_DIR / f"{safe_name}_metadata.txt"
    metadata = "\n".join(
        [
            "barcode_export",
            f"image={image_name}",
            f"model={model_name}",
            f"decoded_text={decoded_text}",
            f"safe_name={safe_name}",
            f"detection_index={det_index}",
            f"detection_confidence={decoded_attempt.detection.confidence:.4f}",
            f"detection_xyxy={[round(v, 1) for v in decoded_attempt.detection.xyxy]}",
            f"scale={decoded_attempt.scale}",
            f"quantile={decoded_attempt.quantile}",
            f"frame_origin={decoded_attempt.frame_origin}",
            f"frame_vx={decoded_attempt.frame_vx}",
            f"frame_vy={decoded_attempt.frame_vy}",
        ]
    )
    metadata_path.write_text(metadata, encoding="utf-8")
    print(metadata)
    return metadata_path


def main() -> None:
    export_decoded_artifacts()


if __name__ == "__main__":
    main()
