import streamlit as st
import pandas as pd

st.title("🏗️ QCNN Architecture")

st.markdown("""
## Hybrid Quantum-Classical Architecture

This project implements a Hybrid Quantum-Classical Neural Network
based on a Quantum Convolutional Neural Network (QCNN).

Classical image features are first projected into a quantum
representation, processed through quantum convolution and
entanglement layers, and finally classified using a
classical output layer.
""")

st.divider()

st.subheader("🔄 Processing Pipeline")

st.markdown("""
1. Input Image (28×28)

2. Flatten Layer (784 Features)

3. Linear Projection (784 → 12)

4. Angle Embedding

5. Quantum Convolution Layer 1

6. Quantum Convolution Layer 2

7. Quantum Pooling

8. Strongly Entangling Layers

9. Pauli-Z Measurements

10. Classical Output Layer

11. Class Prediction
""")

st.divider()

architecture = pd.DataFrame({
    "Layer": [
        "Input Image",
        "Linear Projection",
        "Angle Embedding",
        "Quantum Convolution 1",
        "Quantum Convolution 2",
        "Quantum Pooling",
        "Strongly Entangling Layers",
        "Measurement",
        "Output Layer"
    ],
    "Purpose": [
        "Image Data",
        "Reduce Feature Dimension",
        "Encode Classical Data",
        "Quantum Feature Extraction",
        "Hierarchical Feature Learning",
        "Dimensionality Reduction",
        "Quantum Correlation Learning",
        "Pauli-Z Expectation Values",
        "Classification"
    ]
})

st.dataframe(
    architecture,
    use_container_width=True,
    hide_index=True
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Qubits",
        "12"
    )

with col2:
    st.metric(
        "Trainable Parameters",
        "9,482"
    )

with col3:
    st.metric(
        "Embedding",
        "Angle"
    )

st.divider()

st.subheader("⚛️ Quantum Circuit Design")

st.markdown("""
### Quantum Convolution

The QCNN applies parameterized quantum convolution blocks
using rotation gates and entangling operations.

### Quantum Pooling

Pooling progressively reduces the number of active quantum
states, creating a hierarchical representation similar to
classical CNN architectures.

### Strongly Entangling Layers

Variational quantum layers capture correlations between
qubits and enable trainable quantum feature extraction.

### Measurement

The final quantum state is measured using Pauli-Z expectation
values, producing classical features for the output classifier.
""")

st.divider()

st.success("""
The QCNN uses only 9,482 trainable parameters while incorporating
quantum feature extraction through variational quantum circuits.
""")