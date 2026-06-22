import streamlit as st
import pandas as pd

st.title("⚖️ Model Comparison")

st.markdown("""
## Classical vs Quantum Models

This project compares Artificial Neural Networks (ANN),
Convolutional Neural Networks (CNN), and a Hybrid
Quantum-Classical Neural Network (QCNN).

The focus is on classification performance,
parameter efficiency, and quantum machine learning feasibility.
""")

st.header("🏗️ Architecture Comparison")

comparison = pd.DataFrame({
    "Model": [
        "ANN",
        "CNN",
        "QCNN"
    ],
    "Architecture": [
        "Fully Connected Neural Network",
        "Convolutional Neural Network",
        "Hybrid Quantum-Classical Neural Network"
    ],
    "Parameters": [
        235146,
        421642,
        9482
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.header("⚙️ Parameter Efficiency")

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

reduction_vs_ann = round(
    235146 / 9482,
    1
)

reduction_vs_cnn = round(
    421642 / 9482,
    1
)

st.header("📊 Parameter Reduction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Reduction vs ANN",
        f"{reduction_vs_ann}×"
    )

with col2:
    st.metric(
        "Reduction vs CNN",
        f"{reduction_vs_cnn}×"
    )

st.divider()

st.success(
    "QCNN achieves significant parameter reduction while "
    "maintaining meaningful classification performance."
)

st.info("""
Model Characteristics

ANN
• Dense fully connected architecture
• Strong baseline performance
• Moderate parameter count

CNN
• Convolution-based feature extraction
• Highest classification accuracy
• Largest parameter count

QCNN
• Hybrid quantum-classical architecture
• Variational quantum circuits
• Extremely parameter efficient
• Demonstrates quantum machine learning concepts
""")

st.divider()

st.subheader("Research Objective")

st.markdown("""
The purpose of this project is not necessarily to outperform
classical CNNs, but to investigate whether hybrid quantum-classical
architectures can learn useful representations while using
dramatically fewer trainable parameters.

This study demonstrates that QCNNs can perform meaningful
classification tasks using only **9,482 trainable parameters**
compared with **421,642 parameters** in the CNN baseline.
""")