import streamlit as st
import pandas as pd

st.title("📊 Datasets")

st.markdown("""
This project evaluates classical and quantum machine learning
models on two standard image classification benchmarks:

• MNIST
• Fashion-MNIST

Both datasets contain grayscale images with 10 classes,
allowing direct comparison between ANN, CNN, and QCNN architectures.
""")

st.divider()

dataset = pd.DataFrame({
    "Dataset": [
        "MNIST",
        "Fashion-MNIST"
    ],
    "Training Images": [
        60000,
        60000
    ],
    "Test Images": [
        10000,
        10000
    ],
    "Classes": [
        10,
        10
    ],
    "Image Size": [
        "28×28",
        "28×28"
    ],
    "Type": [
        "Handwritten Digits",
        "Fashion Articles"
    ]
})

st.dataframe(
    dataset,
    use_container_width=True,
    hide_index=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("🔢 MNIST")

    st.markdown("""
**Dataset Purpose**

Handwritten digit recognition benchmark.

**Classes**

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

**Characteristics**

- Simple grayscale images
- Low visual complexity
- Standard computer vision benchmark
- Frequently used in machine learning research
""")

with col2:

    st.subheader("👕 Fashion-MNIST")

    st.markdown("""
**Dataset Purpose**

Fashion article classification benchmark.

**Classes**

T-shirt, Trouser, Pullover, Dress,
Coat, Sandal, Shirt, Sneaker,
Bag, Ankle Boot

**Characteristics**

- Grayscale fashion images
- More challenging than MNIST
- Greater visual similarity between classes
- Better benchmark for feature extraction
""")

st.divider()

st.subheader("📈 Why These Datasets?")

st.markdown("""
The project uses both datasets to evaluate how well
Hybrid Quantum-Classical Neural Networks generalize across
different image classification tasks.

### MNIST

Provides a baseline benchmark and allows comparison with
existing quantum machine learning literature.

### Fashion-MNIST

Introduces more complex visual patterns and class overlap,
making it a stronger test of feature extraction capability.

Using both datasets provides a more balanced evaluation of
classical and quantum learning architectures.
""")

st.divider()

st.success("""
Both datasets contain 70,000 images (60,000 training and
10,000 testing) and are widely used benchmarks for evaluating
machine learning and quantum machine learning models.
""")