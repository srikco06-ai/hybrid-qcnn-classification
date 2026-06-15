import streamlit as st
import pandas as pd

st.title("📊 Dataset")

dataset = pd.DataFrame({
    "Dataset": ["MNIST", "Fashion-MNIST"],
    "Classes": [10, 10],
    "Image Size": ["28×28", "28×28"],
    "Type": [
        "Handwritten Digits",
        "Fashion Articles"
    ]
})

st.dataframe(
    dataset,
    use_container_width=True
)

st.markdown("""
### MNIST

- Handwritten digit recognition
- Benchmark computer vision dataset
- 10 classes

### Fashion-MNIST

- Fashion article classification
- More challenging visual patterns
- 10 classes
""")