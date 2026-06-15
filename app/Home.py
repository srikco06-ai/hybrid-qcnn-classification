import streamlit as st

st.set_page_config(
    page_title="Hybrid QCNN Classification",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hybrid Quantum-Classical Machine Learning")

st.subheader(
    "Quantum Convolutional Neural Network (QCNN)"
)

st.markdown("""
This project investigates Hybrid Quantum-Classical Machine Learning
for image classification using Quantum Convolutional Neural Networks (QCNN).

The study compares:

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Quantum Convolutional Neural Networks (QCNN)

Datasets:

- MNIST
- Fashion-MNIST

Developed as part of the IIT Delhi Quantum Computing and Machine Learning Programme.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("QCNN Accuracy", "82.35%")

with col2:
    st.metric("QCNN Parameters", "9,482")

with col3:
    st.metric("Datasets", "2")

st.divider()

st.success(
    "Use the sidebar to explore the project architecture, datasets, comparisons and results."
)