"""
Phase 2.3 - Live Inference (Version 1)
"""

from pathlib import Path
import sys
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine

st.set_page_config(page_title="Live Inference", page_icon="⚛️", layout="wide")

@st.cache_resource
def get_engine():
    return InferenceEngine()

engine = get_engine()

st.title("⚛️ Live QCNN Image Inference")
st.write("Upload an image, choose a dataset and model, then click Predict.")

left, right = st.columns([1, 2])

with left:
    dataset = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST"])
    model_name = st.selectbox("Model", ["ANN", "CNN", "QCNN"])
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    predict = st.button("🚀 Predict", use_container_width=True)

image = None
with right:
    st.subheader("Image Preview")
    if uploaded:
        image = Image.open(uploaded).convert("L")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to preview.")

st.divider()

if predict:
    if image is None:
        st.warning("Please upload an image.")
    else:
        try:
            with st.spinner("Running inference..."):
                result = engine.predict_image(
                    image=image,
                    model_name=model_name,
                    dataset=dataset,
                )

            a, b, c = st.columns(3)
            a.metric("Prediction", result["label"])
            b.metric("Confidence", f"{result['confidence']:.2f}%")
            c.metric("Inference Time", f"{result['inference_time']:.2f} ms")

            st.subheader("Prediction Details")
            st.json(result, expanded=False)

        except Exception as exc:
            st.exception(exc)

with st.expander("Project Information"):
    st.markdown("""
Version 1 includes:

- Upload image
- Preview image
- Real inference
- Prediction
- Confidence
- Inference time

Next version will add probability charts and Top-3 predictions.
""")
