"""
Phase 2.5
Live Model Comparison Dashboard

Compares ANN, CNN and QCNN predictions
using a single uploaded image.

Backend:
    Frozen
    Uses existing InferenceEngine only.
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
    page_title="Live Model Comparison",
    page_icon="📊",
    layout="wide",
)


# ==========================================================
# Cached Resources
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_engine() -> InferenceEngine:
    return InferenceEngine()


engine = get_engine()


# ==========================================================
# Constants
# ==========================================================

MODELS = [
    "ANN",
    "CNN",
    "QCNN",
]

DATASETS = [
    "MNIST",
    "Fashion-MNIST",
]

MODEL_INFO = {
    "ANN": {
        "Architecture": "Artificial Neural Network",
        "Framework": "PyTorch",
        "Input": "28×28",
        "Output": "10 Classes",
    },
    "CNN": {
        "Architecture": "Convolutional Neural Network",
        "Framework": "PyTorch",
        "Input": "28×28",
        "Output": "10 Classes",
    },
    "QCNN": {
        "Architecture": "Hybrid Quantum CNN",
        "Framework": "PennyLane + PyTorch",
        "Input": "28×28",
        "Output": "10 Classes",
    },
}

DATASET_INFO = {
    "MNIST": {
        "Type": "Handwritten Digits",
        "Classes": "10",
        "Image Size": "28×28",
        "Channels": "Grayscale",
    },
    "Fashion-MNIST": {
        "Type": "Fashion Articles",
        "Classes": "10",
        "Image Size": "28×28",
        "Channels": "Grayscale",
    },
}


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
                round(x * 100, 2)
                for x in probabilities
            ],
        }
    )


def render_info_card(
    title: str,
    info: Dict[str, str],
):

    st.subheader(title)

    df = pd.DataFrame(
        {
            "Property": list(info.keys()),
            "Value": list(info.values()),
        }
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )


def run_model(
    image: Image.Image,
    dataset: str,
    model: str,
) -> Dict:

    return engine.predict_image(
        image=image,
        dataset=dataset,
        model_name=model,
    )


def comparison_dataframe(
    results: Dict[str, Dict],
) -> pd.DataFrame:

    rows = []

    for model_name, result in results.items():

        rows.append(
            {
                "Model": model_name,
                "Prediction": result["label"],
                "Confidence (%)": result["confidence"],
                "Inference Time (ms)": result["inference_time"],
            }
        )

    return pd.DataFrame(rows)


def render_metric_card(
    title: str,
    value: str,
):

    st.metric(
        label=title,
        value=value,
    )


# ==========================================================
# Header
# ==========================================================

st.title("📊 Live Model Comparison")

st.markdown(
    """
Upload **one image** and compare predictions from:

- ANN
- CNN
- QCNN

using the same preprocessing pipeline and pretrained models.
"""
)

st.divider()


# ==========================================================
# Configuration
# ==========================================================

config_col, preview_col = st.columns(
    [1, 2],
    gap="large",
)

with config_col:

    st.subheader("Configuration")

    dataset = st.selectbox(
        "Dataset",
        DATASETS,
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
    )

    compare_button = st.button(
        "🚀 Compare Models",
        use_container_width=True,
        type="primary",
    )

with preview_col:

    st.subheader("Image Preview")

    image = None

    if uploaded is not None:

        image = Image.open(uploaded).convert("L")

        st.image(
            image,
            width=250,
        )

    else:

        st.info(
            "Upload an image to begin comparison."
        )


st.divider()


# ==========================================================
# Static Information
# ==========================================================

left_info, right_info = st.columns(
    2,
    gap="large",
)

with left_info:

    render_info_card(
        "📚 Dataset Information",
        DATASET_INFO[dataset],
    )

with right_info:

    st.subheader("🤖 Available Models")

    available_models = pd.DataFrame(
        {
            "Model": MODELS,
            "Architecture": [
                MODEL_INFO[m]["Architecture"]
                for m in MODELS
            ],
            "Framework": [
                MODEL_INFO[m]["Framework"]
                for m in MODELS
            ],
        }
    )

    st.dataframe(
        available_models,
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ==========================================================
# Comparison Inference
# ==========================================================

if compare_button:

    if image is None:

        st.warning(
            "Please upload an image before running comparison."
        )
        st.stop()

    results: Dict[str, Dict] = {}

    progress = st.progress(0)

    status = st.empty()

    try:

        total_models = len(MODELS)

        for index, model_name in enumerate(MODELS):

            status.info(
                f"Running {model_name} inference..."
            )

            results[model_name] = run_model(
                image=image,
                dataset=dataset,
                model=model_name,
            )

            progress.progress(
                (index + 1) / total_models
            )

        status.success(
            "Inference completed successfully."
        )

    except Exception as exc:

        st.exception(exc)
        st.stop()

    st.divider()

    # ======================================================
    # Comparison Summary
    # ======================================================

    st.header("📊 Model Comparison")

    summary = comparison_dataframe(results)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ======================================================
    # Individual Result Cards
    # ======================================================

    st.header("🏆 Prediction Cards")

    card_columns = st.columns(3)

    for column, model_name in zip(
        card_columns,
        MODELS,
    ):

        result = results[model_name]

        with column:

            st.subheader(model_name)

            render_metric_card(
                "Prediction",
                result["label"],
            )

            render_metric_card(
                "Confidence",
                f"{result['confidence']:.2f}%",
            )

            render_metric_card(
                "Inference Time",
                f"{result['inference_time']:.2f} ms",
            )

    st.divider()

    # ======================================================
    # Probability Distributions
    # ======================================================

    st.header("📈 Probability Distribution")

    probability_columns = st.columns(3)

    for column, model_name in zip(
        probability_columns,
        MODELS,
    ):

        with column:

            st.subheader(model_name)

            probability_df = probability_dataframe(
                results[model_name]["probabilities"],
                dataset,
            )

            st.bar_chart(
                probability_df.set_index("Class"),
                use_container_width=True,
            )

    st.divider()

    # ======================================================
    # Top-3 Predictions
    # ======================================================

    st.header("🥇 Top-3 Predictions")

    top_columns = st.columns(3)

    for column, model_name in zip(
        top_columns,
        MODELS,
    ):

        with column:

            st.subheader(model_name)

            probability_df = probability_dataframe(
                results[model_name]["probabilities"],
                dataset,
            )

            top3 = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False,
                )
                .head(3)
            )

            st.dataframe(
                top3,
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    
        # ======================================================
    # Detailed Probability Tables
    # ======================================================

    st.header("📋 Detailed Probability Tables")

    detail_columns = st.columns(3)

    for column, model_name in zip(
        detail_columns,
        MODELS,
    ):

        with column:

            st.subheader(model_name)

            probability_df = probability_dataframe(
                results[model_name]["probabilities"],
                dataset,
            )

            probability_df = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True,
                height=390,
            )

    st.divider()

    # ======================================================
    # Raw Prediction JSON
    # ======================================================

    st.header("🧾 Raw Prediction Output")

    json_columns = st.columns(3)

    for column, model_name in zip(
        json_columns,
        MODELS,
    ):

        with column:

            with st.expander(
                f"{model_name} Prediction JSON",
                expanded=False,
            ):

                st.json(
                    results[model_name],
                    expanded=False,
                )

    st.divider()

    # ======================================================
    # Best Model Summary
    # ======================================================

    st.header("⭐ Best Result")

    best_model = max(
        results.items(),
        key=lambda item: item[1]["confidence"],
    )

    model_name = best_model[0]
    result = best_model[1]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Winner",
            model_name,
        )

    with metric2:

        st.metric(
            "Prediction",
            result["label"],
        )

    with metric3:

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%",
        )

    with metric4:

        st.metric(
            "Inference Time",
            f"{result['inference_time']:.2f} ms",
        )

    st.success(
        f"Highest confidence prediction produced by **{model_name}**."
    )

# ==========================================================
# Default Dashboard
# ==========================================================

else:

    st.header("📊 Comparison Dashboard")

    summary_columns = st.columns(4)

    with summary_columns[0]:
        st.metric(
            "Models",
            "3",
        )

    with summary_columns[1]:
        st.metric(
            "Datasets",
            "2",
        )

    with summary_columns[2]:
        st.metric(
            "Predictions",
            "--",
        )

    with summary_columns[3]:
        st.metric(
            "Status",
            "Waiting",
        )

    st.divider()

    empty_left, empty_right = st.columns(
        2,
        gap="large",
    )

    with empty_left:

        st.subheader("Workflow")

        st.markdown(
            """
1. Upload an image

2. Select dataset

3. Click **Compare Models**

4. Review predictions

5. Compare confidence

6. Compare inference time
"""
        )

    with empty_right:

        st.subheader("Compared Models")

        st.dataframe(
            available_models,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    
        # ======================================================
    # Placeholder Layout
    # ======================================================

    placeholder_columns = st.columns(3)

    for column, model_name in zip(
        placeholder_columns,
        MODELS,
    ):

        with column:

            st.subheader(model_name)

            st.metric(
                "Prediction",
                "--",
            )

            st.metric(
                "Confidence",
                "--",
            )

            st.metric(
                "Inference Time",
                "--",
            )

            st.info(
                "Run model comparison to view prediction details."
            )

    st.divider()

    st.subheader("Expected Comparison")

    comparison_placeholder = pd.DataFrame(
        {
            "Model": MODELS,
            "Prediction": ["--"] * 3,
            "Confidence (%)": ["--"] * 3,
            "Inference Time (ms)": ["--"] * 3,
        }
    )

    st.dataframe(
        comparison_placeholder,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Probability Distribution")

    st.info(
        "Each model will display its probability distribution after comparison."
    )

    st.divider()

    st.subheader("Top-3 Predictions")

    info_cols = st.columns(3)

    for column, model_name in zip(info_cols, MODELS):

        with column:

            st.info(
                f"{model_name} Top-3 predictions will appear here."
            )

    st.divider()

    st.subheader("Prediction JSON")

    json_cols = st.columns(3)

    for column, model_name in zip(json_cols, MODELS):

        with column:

            with st.expander(
                f"{model_name} JSON",
                expanded=False,
            ):

                st.info(
                    "Prediction output will appear after running inference."
                )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    """
Hybrid QCNN Classification • Live Model Comparison Dashboard

Compare ANN, CNN and QCNN predictions using a single uploaded image.
The dashboard uses the shared InferenceEngine and identical preprocessing
pipeline to ensure fair comparison across all models.
"""
)