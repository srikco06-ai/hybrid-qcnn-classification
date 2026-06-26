import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Live Inference",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Live QCNN Image Inference")

st.markdown(
    """
Run inference using the trained models included in this repository.

Choose a dataset, select a model, upload an image, and proceed to prediction.
"""
)

st.divider()

left, right = st.columns([1, 2])

with left:

    st.subheader("Configuration")

    dataset = st.selectbox(
        "Dataset",
        [
            "MNIST",
            "Fashion-MNIST",
        ],
    )

    model_name = st.selectbox(
        "Model",
        [
            "ANN",
            "CNN",
            "QCNN",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
    )

with right:

    st.subheader("Image Preview")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True,
        )

    else:

        st.info(
            "Upload an image to preview it here."
        )

st.divider()

st.subheader("Selected Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Dataset",
        value=dataset,
    )

with col2:
    st.metric(
        label="Model",
        value=model_name,
    )

st.divider()

with st.expander("Project Information", expanded=False):

    st.markdown(
        """
### Hybrid Quantum-Classical Image Classification

This application performs image classification using pretrained models.

Available models:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Quantum Convolutional Neural Network (QCNN)

The models were trained on:

- MNIST
- Fashion-MNIST

Phase 2 will connect this interface to the reusable inference engine.
"""
    )

st.success(
    "Phase 2.1 complete: User interface initialized."
)