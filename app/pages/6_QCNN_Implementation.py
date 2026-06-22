import streamlit as st
from pathlib import Path

st.title("⚛️ QCNN Implementation")

st.markdown("""
## Hybrid Quantum-Classical Neural Network

This project implements a Quantum Convolutional Neural Network (QCNN)
using PennyLane and PyTorch.

The architecture combines classical neural network layers with
parameterized quantum circuits to perform image classification on
MNIST and Fashion-MNIST datasets.

The implementation follows a hybrid quantum-classical workflow where
classical features are projected into a quantum latent space,
processed through quantum convolution and pooling operations,
and mapped back to classical outputs for classification.
""")

st.divider()

st.subheader("📌 Model Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Qubits",
        "12"
    )

with col2:
    st.metric(
        "Quantum Params",
        "22"
    )

with col3:
    st.metric(
        "Total Params",
        "9,482"
    )

with col4:
    st.metric(
        "Framework",
        "PennyLane"
    )

st.info("""
Parameter Breakdown

Quantum Parameters
• Quantum Convolution Layer 1 = 2
• Quantum Convolution Layer 2 = 2
• Strongly Entangling Layers = 18

Total Quantum Parameters = 22

Classical Parameters
• Input Projection Layer
• Output Classification Layer

Total Hybrid Model Parameters = 9,482
""")

st.divider()

st.subheader("🔄 QCNN Processing Pipeline")

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

st.subheader("🏗️ Architecture Flow")

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

st.divider()

st.subheader("⚛️ Quantum Circuit Components")

st.markdown("""
### Angle Embedding

Classical image features are encoded into quantum states using
rotation-based angle embedding.

### Quantum Convolution

Parameterized quantum convolution blocks perform local feature
extraction using trainable rotation gates and entanglement.

### Quantum Pooling

Pooling progressively reduces active quantum states, creating
hierarchical representations similar to classical CNNs.

### Strongly Entangling Layers

Variational quantum layers learn correlations between qubits
through trainable entangling operations.

### Measurement

Pauli-Z expectation values convert quantum information into
classical features for final classification.
""")

st.divider()

st.subheader("💡 Why QCNN?")

st.markdown("""
The goal of this project is not necessarily to outperform
classical CNNs.

Instead, the objective is to investigate whether quantum
machine learning models can achieve meaningful classification
performance while using dramatically fewer trainable parameters.

Results demonstrate that the QCNN can learn useful image
representations while operating with only 9,482 trainable
parameters compared to 421,642 parameters in the CNN baseline.
""")

st.divider()

st.subheader("📂 QCNN Source Code")

qcnn_file = Path("src/quantum/qcnn.py")

if qcnn_file.exists():
    with st.expander("View qcnn.py"):
        st.code(
            qcnn_file.read_text(),
            language="python"
        )
else:
    st.warning("qcnn.py not found")

st.divider()

st.subheader("📂 Quantum Circuit Implementation")

circuit_file = Path("src/quantum/circuits.py")

if circuit_file.exists():
    with st.expander("View circuits.py"):
        st.code(
            circuit_file.read_text(),
            language="python"
        )
else:
    st.warning("circuits.py not found")

st.divider()

st.subheader("📂 Inference Module")

infer_file = Path("src/quantum/inference.py")

if infer_file.exists():
    with st.expander("View inference.py"):
        st.code(
            infer_file.read_text(),
            language="python"
        )
else:
    st.warning("inference.py not found")

st.divider()

st.success("""
This implementation demonstrates a complete Hybrid
Quantum-Classical Machine Learning workflow using
PennyLane and PyTorch, including quantum feature
encoding, variational quantum circuits, quantum
convolution, quantum pooling, and image classification.
""")