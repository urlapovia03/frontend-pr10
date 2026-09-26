import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = "https://backend-pr10.onrender.com/predict"
CANVAS_SIZE = 280
MAX_SEND_SIZE = 512  # уменьшаем перед отправкой, чтобы не гонять по сети лишний вес

st.set_page_config(page_title="Классификация животных", page_icon="🐾", layout="centered")
st.title("🐾 Классификация изображений животных")
st.caption("Backend: EfficientNetB4 · классы: butterfly, cow, elephant, sheep, squirrel")

tab_upload, tab_draw = st.tabs(["📤 Загрузить изображение", "✏️ Нарисовать"])

raw_image = None  # сюда попадёт PIL.Image из одного из двух источников

with tab_upload:
    uploaded_file = st.file_uploader("Выбери изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Загруженное изображение", width=280)

with tab_draw:
    st.write("Нарисуй что-нибудь на холсте (белым по чёрному фону):")
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas",
    )
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
        raw_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")

st.divider()


def preprocess_for_sending(img: Image.Image) -> bytes:
    """
    Базовая предобработка перед отправкой на API:
    - приводим к RGB (снимаем альфа-канал, если есть, на белый фон)
    - уменьшаем, если изображение слишком большое (экономим трафик)
    Финальный resize под точный вход модели (224×224) и нормализация
    делаются уже на бэкенде — см. main.py, preprocess_image().
    """
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    if max(img.size) > MAX_SEND_SIZE:
        img.thumbnail((MAX_SEND_SIZE, MAX_SEND_SIZE))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if raw_image is not None:
    if st.button("Классифицировать", type="primary"):
        with st.spinner("Отправляю на сервер..."):
            image_bytes = preprocess_for_sending(raw_image)
            try:
                response = requests.post(
                    API_URL,
                    files={"file": ("image.png", image_bytes, "image/png")},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка запроса к API: {e}")
                result = None

        if result is not None:
            st.success(
                f"Предсказанный класс: **{result['predicted_class']}** "
                f"(уверенность: {result['confidence']:.1%})"
            )

            probs = result["probabilities"]
            df_probs = pd.DataFrame(
                {"Класс": list(probs.keys()), "Вероятность": list(probs.values())}
            ).sort_values("Вероятность", ascending=False)

            st.bar_chart(df_probs.set_index("Класс"))
            st.dataframe(df_probs, hide_index=True, use_container_width=True)
else:
    st.info("Загрузи изображение или нарисуй его на холсте, чтобы начать.")
