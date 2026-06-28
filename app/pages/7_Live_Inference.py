"""
Production Live Inference Dashboard

Hybrid QCNN Classification

Features
--------
- Cached inference engine
- Responsive dashboard
- KPI metrics
- Probability visualization
- Top-3 predictions
- Model information
- Dataset information
- Prediction JSON viewer
- Robust error handling
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine
from src.inference.labels import get_labels


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Live Inference",
    page_icon="⚛️",
    layout="wide",
)


# ==========================================================
# Constants
# ==========================================================

DATASETS = (
    "MNIST",
    "Fashion-MNIST",
)

MODELS = (
    "ANN",
    "CNN",
    "QCNN",
)

MODEL_INFO: Dict[str, Dict[str, str]] = {
    "ANN": {
        "Architecture": "Artificial Neural Network",
        "Framework": "PyTorch",
        "Input": "28 × 28 grayscale",
        "Output": "10 classes",
    },
    "CNN": {
        "Architecture": "Convolutional Neural Network",
        "Framework": "PyTorch",
        "Input": "28 × 28 grayscale",
        "Output": "10 classes",
    },
    "QCNN": {
        "Architecture": "Hybrid Quantum Convolutional Neural Network",
        "Framework": "PennyLane + PyTorch",
        "Input": "28 × 28 grayscale",
        "Output": "10 classes",
    },
}

DATASET_INFO: Dict[str, Dict[str, str]] = {
    "MNIST": {
        "Classes": "10",
        "Image Size": "28 × 28",
        "Channels": "Grayscale",
        "Domain": "Handwritten Digits",
    },
    "Fashion-MNIST": {
        "Classes": "10",
        "Image Size": "28 × 28",
        "Channels": "Grayscale",
        "Domain": "Fashion Apparel",
    },
}


# ==========================================================
# Cached Resources
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_engine() -> InferenceEngine:
    return InferenceEngine()


engine = get_engine()


# ==========================================================
# Helper Functions
# ==========================================================

def probability_dataframe(
    probabilities: List[float],
    dataset: str,
) -> pd.DataFrame:
    labels = get_labels(dataset)

    return pd.DataFrame(
        {
            "Class": [
                labels[i]
                for i in range(len(probabilities))
            ],
            "Probability (%)": [
                round(p * 100, 2)
                for p in probabilities
            ],
        }
    )


def render_information(
    title: str,
    info: Dict[str, str],
) -> None:
    st.subheader(title)

    table = pd.DataFrame(
        {
            "Property": list(info.keys()),
            "Value": list(info.values()),
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


def render_probability_chart(
    dataframe: pd.DataFrame,
) -> None:
    st.bar_chart(
        dataframe.set_index("Class"),
        use_container_width=True,
    )


def render_top_predictions(
    dataframe: pd.DataFrame,
) -> None:
    st.dataframe(
        dataframe.sort_values(
            "Probability (%)",
            ascending=False,
        ).head(3),
        use_container_width=True,
        hide_index=True,
    )


def render_metrics(
    result: Dict,
) -> None:

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Prediction",
        result["label"],
    )

    c2.metric(
        "Confidence",
        f'{result["confidence"]:.2f}%',
    )

    c3.metric(
        "Inference Time",
        f'{result["inference_time"]:.2f} ms',
    )

    c4.metric(
        "Model",
        result["model"],
    )

    c5.metric(
        "Dataset",
        result["dataset"],
    )


# ==========================================================
# Header
# ==========================================================

st.title("⚛️ Live QCNN Image Inference")

st.markdown(
    """
Run inference using pretrained **ANN**, **CNN**, and **QCNN**
models.

Upload an image, choose a dataset and model,
then execute inference to compare predictions.
"""
)

st.divider()


# ==========================================================
# Configuration
# ==========================================================

left, right = st.columns(
    [1, 2],
    gap="large",
)

with left:

    st.subheader("Configuration")

    selected_dataset = st.selectbox(
        "Dataset",
        DATASETS,
    )

    selected_model = st.selectbox(
        "Model",
        MODELS,
    )

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
    )

    predict_button = st.button(
        "🚀 Run Inference",
        use_container_width=True,
        type="primary",
    )

with right:

    st.subheader("Image Preview")

    image = None

    if uploaded_image is not None:

        image = (
            Image.open(uploaded_image)
            .convert("L")
        )

        st.image(
            image,
            use_container_width=True,
        )

    else:

        st.info(
            "Upload a grayscale or RGB image for inference."
        )
        
        # ==========================================================
# Static Information
# ==========================================================

st.divider()

info_col1, info_col2 = st.columns(
    2,
    gap="large",
)

with info_col1:
    render_information(
        "🤖 Model Information",
        MODEL_INFO[selected_model],
    )

with info_col2:
    render_information(
        "📚 Dataset Information",
        DATASET_INFO[selected_dataset],
    )


# ==========================================================
# Inference
# ==========================================================

if predict_button:

    if image is None:

        st.warning(
            "Please upload an image before running inference."
        )
        st.stop()

    try:

        with st.spinner(
            "Running inference..."
        ):

            result = engine.predict_image(
                image=image,
                model_name=selected_model,
                dataset=selected_dataset,
            )

        probability_df = probability_dataframe(
            result["probabilities"],
            selected_dataset,
        )

    except Exception as exc:

        st.exception(exc)
        st.stop()

    # ======================================================
    # Metrics
    # ======================================================

    st.divider()

    st.subheader("📊 Prediction Summary")

    render_metrics(result)

    # ======================================================
    # Charts
    # ======================================================

    st.divider()

    chart_col, top3_col = st.columns(
        2,
        gap="large",
    )

    with chart_col:

        st.subheader(
            "📈 Probability Distribution"
        )

        render_probability_chart(
            probability_df,
        )

    with top3_col:

        st.subheader(
            "🏆 Top-3 Predictions"
        )

        render_top_predictions(
            probability_df,
        )

    # ======================================================
    # Prediction Details
    # ======================================================

    st.divider()

    details_left, details_right = st.columns(
        [2, 1],
        gap="large",
    )

    with details_left:

        st.subheader(
            "Prediction Details"
        )

        styled = (
            probability_df
            .sort_values(
                "Probability (%)",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )

    with details_right:

        st.subheader(
            "Prediction JSON"
        )

        st.json(
            result,
            expanded=False,
        )
        
        # ==========================================================
# Empty Dashboard State
# ==========================================================

else:

    st.divider()

    st.subheader("📊 Prediction Dashboard")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    metric1.metric(
        "Prediction",
        "--",
    )

    metric2.metric(
        "Confidence",
        "--",
    )

    metric3.metric(
        "Inference Time",
        "--",
    )

    metric4.metric(
        "Model",
        "--",
    )

    metric5.metric(
        "Dataset",
        "--",
    )

    st.divider()

    left_panel, right_panel = st.columns(
        2,
        gap="large",
    )

    with left_panel:

        st.subheader(
            "📈 Probability Distribution"
        )

        st.info(
            "Run inference to visualize class probabilities."
        )

    with right_panel:

        st.subheader(
            "🏆 Top-3 Predictions"
        )

        st.info(
            "The three most probable classes will be displayed here."
        )

    st.divider()

    st.subheader("Prediction JSON")

    st.info(
        "Prediction metadata and raw inference output will appear here after a successful prediction."
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    (
        "Hybrid QCNN Classification • "
        "Interactive Live Inference Dashboard • "
        "Supports ANN, CNN and QCNN models for "
        "MNIST and Fashion-MNIST datasets."
    )
)