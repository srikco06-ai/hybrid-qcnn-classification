import streamlit as st
import pandas as pd

st.title("⚖️ Model Comparison")

st.markdown("""
## Classical vs Quantum Models

This project compares traditional deep learning models with a
Hybrid Quantum-Classical Neural Network (QCNN).

The objective is to evaluate classification performance,
parameter efficiency, and feasibility of quantum machine
learning architectures.
""")

comparison = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Parameters": [235146, 421642, 9482],
    "Architecture": [
        "Fully Connected Neural Network",
        "Convolutional Neural Network",
        "Hybrid Quantum-Classical Neural Network"
    ],
    "MNIST Accuracy": [
        "98.00%",
        "98.81%",
        "82.35%"
    ],
    "Fashion-MNIST Accuracy": [
        "88.74%",
        "92.06%",
        "59.35%"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Parameter Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ANN Parameters",
        "235,146"
    )

with col2:
    st.metric(
        "CNN Parameters",
        "421,642"
    )

with col3:
    st.metric(
        "QCNN Parameters",
        "9,482"
    )

st.divider()

st.success(
    "QCNN achieved substantial parameter reduction compared with "
    "classical ANN and CNN architectures."
)

st.info(
    "Although CNN achieved the highest classification accuracy, "
    "QCNN demonstrates the feasibility of hybrid quantum-classical "
    "learning using variational quantum circuits."
)