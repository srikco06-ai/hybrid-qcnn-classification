import streamlit as st

st.set_page_config(
    page_title="Hybrid QCNN Classification",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hybrid Quantum-Classical Machine Learning")

st.subheader(
    "Quantum Convolutional Neural Networks for Image Classification"
)

st.markdown("""
This project investigates Hybrid Quantum-Classical Machine Learning
using Quantum Convolutional Neural Networks (QCNNs) implemented with
PennyLane and PyTorch.

The work originated from the IIT Delhi Continuing Education Programme
in Quantum Computing and Machine Learning and has been extended into a
complete research reproduction and engineering implementation.

The project evaluates whether QCNN architectures can perform image
classification while using significantly fewer trainable parameters
than traditional deep learning models.
""")

st.divider()

st.header("🎯 Models Evaluated")

st.markdown("""
The study compares three architectures:

• Artificial Neural Network (ANN)

• Convolutional Neural Network (CNN)

• Hybrid Quantum-Classical Neural Network (QCNN)
""")

st.divider()

st.header("📊 Datasets")

st.markdown("""
Two standard computer vision benchmarks are used:

• MNIST (Handwritten Digits)

• Fashion-MNIST (Fashion Article Classification)

Each dataset contains:

- 60,000 training images
- 10,000 testing images
- 10 output classes
- 28×28 grayscale images
""")

st.divider()

st.header("🚀 Reproduced Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MNIST QCNN",
        "82.31%"
    )

with col2:
    st.metric(
        "Fashion QCNN",
        "78.44%"
    )

with col3:
    st.metric(
        "QCNN Parameters",
        "9,482"
    )

st.divider()

st.header("⚙️ Parameter Efficiency")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ANN",
        "235,146"
    )

with col2:
    st.metric(
        "CNN",
        "421,642"
    )

with col3:
    st.metric(
        "QCNN",
        "9,482"
    )

st.success(
    "QCNN achieves substantial parameter reduction while maintaining meaningful classification performance."
)

st.divider()

st.header("🔬 Research Question")

st.markdown("""
Can Quantum Convolutional Neural Networks achieve useful image
classification performance while using dramatically fewer trainable
parameters than classical neural network architectures?
""")

st.divider()

st.header("📚 Project Sections")

st.markdown("""
Use the sidebar to explore:

- Project Overview
- QCNN Architecture
- Dataset Analysis
- Model Comparison
- Experimental Results
- QCNN Implementation

The implementation includes classical baselines, quantum circuits,
training pipelines, evaluation metrics, and source-code inspection.
""")

st.divider()

st.info("""
This repository demonstrates a complete Hybrid Quantum-Classical
Machine Learning workflow including model design, quantum circuit
construction, training, evaluation, reproduction, and deployment.
""")