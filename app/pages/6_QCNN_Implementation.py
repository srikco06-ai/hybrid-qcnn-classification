import streamlit as st
from pathlib import Path

st.title("⚛️ QCNN Implementation")

st.markdown("""
## Actual Quantum Machine Learning Implementation

This page documents the Hybrid Quantum-Classical Neural Network
used in this project.

The QCNN combines classical preprocessing with quantum feature
extraction using PennyLane and PyTorch.

### Technologies

- PennyLane
- PyTorch
- Variational Quantum Circuits (VQC)
- Angle Embedding
- Quantum Convolution
- Quantum Pooling
- Strongly Entangling Layers
""")

st.divider()

st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Qubits", "12")

with col2:
    st.metric("Parameters", "9,482")

with col3:
    st.metric("Framework", "PennyLane")

st.divider()

st.subheader("Quantum Processing Pipeline")

st.markdown("""
1. Input Image (28×28)

2. Flatten Layer (784 Features)

3. Linear Projection

4. Angle Embedding

5. Quantum Convolution Layer

6. Quantum Pooling Layer

7. Strongly Entangling Layers

8. Pauli-Z Measurements

9. Classical Classification Layer

10. Final Prediction
""")

st.divider()

st.subheader("QCNN Architecture")

st.code("""
Input Image (28×28)

        ↓

Flatten (784 Features)

        ↓

Linear Projection

        ↓

12-Qubit Angle Embedding

        ↓

Quantum Convolution Layer 1

        ↓

Quantum Convolution Layer 2

        ↓

Quantum Pooling

        ↓

Strongly Entangling Layers

        ↓

Pauli-Z Measurements

        ↓

Output Layer

        ↓

Class Prediction
""")

st.info("""
The QCNN did not outperform classical CNNs in accuracy.

The purpose of this project was to investigate Hybrid
Quantum-Classical Learning using Variational Quantum Circuits
and Quantum Convolutional Neural Networks.

The results demonstrate that QCNNs can successfully learn
image representations while operating with quantum layers,
highlighting their potential for future NISQ-era quantum
computing applications.
""")

st.divider()

st.subheader("QCNN Source Code")

qcnn_file = Path("src/quantum/qcnn.py")

if qcnn_file.exists():
    with st.expander("View QCNN Source Code"):
        st.code(
            qcnn_file.read_text(),
            language="python"
        )
else:
    st.warning("qcnn.py not found")

st.divider()

st.subheader("Quantum Circuit Implementation")

circuit_file = Path("src/quantum/circuits.py")

if circuit_file.exists():
    with st.expander("View Quantum Circuit Code"):
        st.code(
            circuit_file.read_text(),
            language="python"
        )
else:
    st.warning("circuits.py not found")

st.divider()

st.subheader("Inference Module")

infer_file = Path("src/quantum/inference.py")

if infer_file.exists():
    with st.expander("View Inference Code"):
        st.code(
            infer_file.read_text(),
            language="python"
        )
else:
    st.warning("inference.py not found")

st.divider()

st.success("""
This implementation demonstrates a complete hybrid
quantum-classical workflow using PennyLane and PyTorch,
including quantum feature extraction, variational quantum
circuits, and image classification.
""")