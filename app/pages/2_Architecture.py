import streamlit as st
import pandas as pd

st.title("🏗️ QCNN Architecture")

st.markdown("""
## Hybrid Quantum-Classical Pipeline

The QCNN architecture combines classical feature processing
with variational quantum circuits.

### Workflow

1. Input Image
2. Feature Encoding
3. Angle Embedding
4. Quantum Convolution
5. Quantum Pooling
6. Measurement
7. Classification
""")

architecture = pd.DataFrame({
    "Layer": [
        "Input",
        "Angle Embedding",
        "Quantum Convolution",
        "Quantum Pooling",
        "Measurement",
        "Output"
    ],
    "Purpose": [
        "Image Features",
        "Encode Classical Data",
        "Quantum Feature Extraction",
        "Reduce Dimensionality",
        "Read Quantum State",
        "Prediction"
    ]
})

st.dataframe(
    architecture,
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

col1.metric("Qubits", "4")
col2.metric("VQC Layers", "2")
col3.metric("Embedding", "Angle")