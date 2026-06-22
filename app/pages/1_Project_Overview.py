import streamlit as st

st.title("📖 Project Overview")

st.markdown("""
## Project Background

This project was developed as part of the IIT Delhi Continuing
Education Programme in Quantum Computing and Machine Learning.

The objective was to investigate whether Hybrid Quantum-Classical
Neural Networks can perform image classification using significantly
fewer trainable parameters than traditional deep learning models.

The study focuses on Quantum Convolutional Neural Networks (QCNNs)
implemented using PennyLane and PyTorch and evaluated on standard
computer vision benchmarks.
""")

st.divider()

st.header("🎯 Research Motivation")

st.markdown("""
Modern deep learning models often require hundreds of thousands
or millions of trainable parameters.

Quantum Machine Learning explores whether parameterized quantum
circuits can learn useful representations while operating with
dramatically smaller model sizes.

Quantum Convolutional Neural Networks (QCNNs) adapt ideas from
classical convolutional neural networks to quantum computing
systems by combining:

- Quantum feature encoding
- Variational quantum circuits
- Quantum convolution operations
- Quantum pooling strategies
- Hybrid quantum-classical learning
""")

st.divider()

st.header("🎯 Project Objectives")

st.markdown("""
### Core Objectives

- Design a Hybrid Quantum-Classical Architecture
- Implement a Quantum Convolutional Neural Network
- Benchmark against ANN and CNN baselines
- Evaluate classification performance
- Study parameter efficiency
- Investigate practical quantum machine learning workflows

### Engineering Objectives

- Reproduce original experimental results
- Build a standalone PennyLane implementation
- Create an interactive Streamlit dashboard
- Document the complete research workflow
""")

st.divider()

st.header("❓ Research Question")

st.markdown("""
Can Quantum Convolutional Neural Networks achieve meaningful
classification performance while using substantially fewer
trainable parameters than classical neural networks?
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Datasets",
        "2"
    )

with col2:
    st.metric(
        "Models",
        "3"
    )

with col3:
    st.metric(
        "QCNN Parameters",
        "9,482"
    )

st.divider()

st.header("🔬 Study Components")

st.markdown("""
### Original Academic Study

The original project compared ANN, CNN, and QCNN architectures
on MNIST and Fashion-MNIST datasets.

### Reproduction Study

The repository reproduces the original architecture and
experimental workflow using a standalone implementation.

### Future Work

Potential extensions include:

- Deeper quantum circuits
- Alternative embedding methods
- Improved pooling strategies
- Quantum Vision Transformers
- Noise-aware quantum simulations
- Real quantum hardware benchmarking
""")

st.divider()

st.success("""
This project demonstrates a complete Quantum Machine Learning
workflow including model design, training, evaluation,
reproduction, and deployment using PennyLane and PyTorch.
""")