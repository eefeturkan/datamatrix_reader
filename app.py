from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import time

import cv2
from PIL import Image
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datamatrix_reader import decode_image as legacy_decode_image
from experiments.step13_od_then_range_grid_decode import run as od_range_decode


MODEL_PATH = ROOT / "best.pt"
UI_UPLOAD_DIR = ROOT / "artifacts" / "ui_uploads"
OD_OUT_DIR = ROOT / "artifacts" / "od_range_decode"


st.set_page_config(page_title="Data Matrix Reader", layout="wide")
st.title("Data Matrix Reader")
st.caption(
    "OD ile Data Matrix bolgesini bulur, crop icinde 20x20 grid fit eder, "
    "local range tabanli bit matrisi cikarip zxingcpp ile okur."
)


def save_uploaded_file(uploaded_file) -> Path:
    UI_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = UI_UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
    target.write_bytes(uploaded_file.getvalue())
    return target


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Dosya bulunamadi: {path}")


def show_text_file(path: Path) -> None:
    if path.exists():
        st.code(path.read_text(encoding="utf-8"), language="text")


uploaded_file = st.file_uploader(
    "PNG/JPG/BMP/WebP gorsel yukleyin",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Baslamak icin bir gorsel yukleyin. Tam gorsel icin `saf.png` iyi bir testtir.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

left, right = st.columns([1, 1])
with left:
    st.subheader("Yuklenen Gorsel")
    st.image(original_image, use_container_width=True)
with right:
    st.subheader("Model ve Akis")
    st.write(f"Model: `{MODEL_PATH.name}`")
    st.write("Ana akis: OD -> crop -> grid fit -> local range -> bit matrix -> decode")
    if not MODEL_PATH.exists():
        st.error("`best.pt` bulunamadi. OD akisi calisamaz.")

tab_od, tab_legacy = st.tabs(["OD + Range Grid", "Eski Decoder"])

with tab_od:
    st.subheader("Son Hal: OD + Range Grid Decode")
    confidence = st.slider("OD confidence threshold", 0.05, 0.90, 0.15, 0.05)
    max_detections = st.slider("Denenen maksimum bbox sayisi", 1, 10, 3, 1)

    if st.button("OD ile Tara ve Oku", type="primary"):
        image_path = save_uploaded_file(uploaded_file)
        with st.spinner("Model bbox buluyor, crop uzerinde range-grid decode calisiyor..."):
            attempts = od_range_decode(
                image_path=image_path,
                model_path=MODEL_PATH,
                confidence=confidence,
                max_detections=max_detections,
            )

        stem = image_path.stem
        summary_path = OD_OUT_DIR / f"{stem}_summary.txt"
        detections_overlay = OD_OUT_DIR / f"{stem}_detections_overlay.png"

        st.subheader("Sonuc")
        decoded_attempts = [attempt for attempt in attempts if attempt.decoded]
        if decoded_attempts:
            st.success(decoded_attempts[0].decoded[0])
        else:
            st.warning("Decode sonucu bulunamadi. Asagidaki ara ciktlari kontrol edin.")

        show_text_file(summary_path)

        st.subheader("1. OD Detection")
        show_image_if_exists(detections_overlay, "Modelin buldugu bbox'lar")

        if attempts:
            first = decoded_attempts[0] if decoded_attempts else attempts[0]
            det_index = first.detection.index
            crop_path = OD_OUT_DIR / f"{stem}_det{det_index}_crop.png"
            pipeline_path = OD_OUT_DIR / f"{stem}_det{det_index}_pipeline.png"
            grid_path = OD_OUT_DIR / f"{stem}_det{det_index}_grid_overlay.png"
            response_path = OD_OUT_DIR / f"{stem}_det{det_index}_range_response.png"

            st.subheader("2. Pipeline Asamalari")
            show_image_if_exists(pipeline_path, "OD crop -> fitted grid -> local range -> rendered bits")

            cols = st.columns(3)
            with cols[0]:
                show_image_if_exists(crop_path, "Crop")
            with cols[1]:
                show_image_if_exists(grid_path, "20x20 grid fit")
            with cols[2]:
                show_image_if_exists(response_path, "Local range response")

            st.subheader("Aday Detaylari")
            st.dataframe(
                [
                    {
                        "det": attempt.detection.index,
                        "conf": round(attempt.detection.confidence, 4),
                        "bbox": [round(v, 1) for v in attempt.detection.xyxy],
                        "scale": attempt.scale,
                        "quantile": attempt.quantile,
                        "decoded": " | ".join(attempt.decoded),
                    }
                    for attempt in attempts
                ],
                use_container_width=True,
                hide_index=True,
            )

with tab_legacy:
    st.subheader("Eski Genel Decoder")
    st.caption("Bu sekme onceki pipeline'i korur; OD kullanmaz.")

    if st.button("Eski Decoder ile Tara"):
        with st.spinner("Eski decoder adaylari deneniyor..."):
            result = legacy_decode_image(image_bytes)

        st.subheader("Sonuc")
        if result.text:
            st.success(result.text)
        else:
            st.warning("Data Matrix okunamadi.")

        if result.processed_image is not None:
            processed_rgb = cv2.cvtColor(result.processed_image, cv2.COLOR_GRAY2RGB)
            st.image(processed_rgb, caption="Kullanilan islenmis aday", use_container_width=True)

        cols = st.columns(3)
        cols[0].metric("Engine", result.engine or "-")
        cols[1].metric("Asama", result.stage_name)
        cols[2].metric("Skor", f"{result.score:.2f}")

        if result.alternatives:
            st.subheader("Alternatif Adaylar")
            st.dataframe(
                [
                    {
                        "text": item.text,
                        "engine": item.engine,
                        "stage": item.stage_name,
                        "score": item.score,
                    }
                    for item in result.alternatives
                ],
                use_container_width=True,
                hide_index=True,
            )
