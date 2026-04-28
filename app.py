from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datamatrix_reader import decode_image


st.set_page_config(page_title="Data Matrix Reader", layout="wide")
st.title("Data Matrix Reader")
st.caption("Data Matrix okumayı çok-aşamalı ön işleme ve iki decoder ile dener.")

uploaded_file = st.file_uploader(
    "PNG/JPG görsel yükleyin",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Orijinal Görsel")
        st.image(original_image, use_container_width=True)

    if st.button("Tara", type="primary"):
        with st.spinner("Farklı ön işleme yaklaşımları deneniyor..."):
            result = decode_image(image_bytes)

        processed_rgb = cv2.cvtColor(result.processed_image, cv2.COLOR_GRAY2RGB)

        with right:
            st.subheader("Kullanılan İşlenmiş Görsel")
            st.image(processed_rgb, use_container_width=True)

        st.subheader("Sonuç")
        if result.text:
            st.success(result.text)
        else:
            st.warning("Data Matrix okunamadı. En iyi aday işlenmiş görsel yukarıda gösteriliyor.")

        meta_left, meta_right, meta_third = st.columns(3)
        meta_left.metric("Engine", result.engine or "-")
        meta_right.metric("Aşama", result.stage_name)
        meta_third.metric("Skor", f"{result.score:.2f}")

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
else:
    st.info("Başlamak için bir görsel yükleyin. İlk denemeler için `test4.png` ve `cropped.png` iyi adaylar.")
