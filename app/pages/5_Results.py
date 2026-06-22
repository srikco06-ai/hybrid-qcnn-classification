import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Experimental Results")

st.markdown("""
This page presents both:

1. Original IIT Delhi QCNN project results
2. Reproduced repository results

The reproduction study validates the original implementation
using a standalone PennyLane + PyTorch codebase.
""")

st.divider()

# =========================
# ORIGINAL RESULTS
# =========================

st.header("🎓 Original IIT Project Results")

mnist_original = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [98.00, 98.81, 82.35]
})

fashion_original = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [88.74, 92.06, 59.35]
})

tab1, tab2 = st.tabs(
    ["MNIST (Original)", "Fashion-MNIST (Original)"]
)

with tab1:

    fig = px.bar(
        mnist_original,
        x="Model",
        y="Accuracy",
        title="Original MNIST Results"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    fig = px.bar(
        fashion_original,
        x="Model",
        y="Accuracy",
        title="Original Fashion-MNIST Results"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# =========================
# REPRODUCED RESULTS
# =========================

st.header("🔬 Reproduced Results")

mnist_reproduced = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [97.49, 99.12, 82.31]
})

fashion_reproduced = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [87.34, 91.59, 78.44]
})

tab3, tab4 = st.tabs(
    ["MNIST (Reproduced)", "Fashion-MNIST (Reproduced)"]
)

with tab3:

    fig = px.bar(
        mnist_reproduced,
        x="Model",
        y="Accuracy",
        title="Reproduced MNIST Results"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab4:

    fig = px.bar(
        fashion_reproduced,
        x="Model",
        y="Accuracy",
        title="Reproduced Fashion-MNIST Results"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# =========================
# COMPARISON TABLE
# =========================

st.header("📋 Original vs Reproduced")

comparison = pd.DataFrame({
    "Model": [
        "ANN",
        "CNN",
        "QCNN",
        "ANN",
        "CNN",
        "QCNN"
    ],
    "Dataset": [
        "MNIST",
        "MNIST",
        "MNIST",
        "Fashion-MNIST",
        "Fashion-MNIST",
        "Fashion-MNIST"
    ],
    "Original": [
        98.00,
        98.81,
        82.35,
        88.74,
        92.06,
        59.35
    ],
    "Reproduced": [
        97.49,
        99.12,
        82.31,
        87.34,
        91.59,
        78.44
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =========================
# PARAMETER COMPARISON
# =========================

st.header("⚙️ Parameter Efficiency")

parameters = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Parameters": [
        235146,
        421642,
        9482
    ]
})

fig = px.bar(
    parameters,
    x="Model",
    y="Parameters",
    title="Trainable Parameter Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

st.success(
    "QCNN uses only 9,482 trainable parameters compared with "
    "235,146 for ANN and 421,642 for CNN."
)

st.info("""
Key Findings

• QCNN reproduced MNIST performance within 0.04% of the original study.

• CNN remains the highest-performing model on both datasets.

• QCNN achieved substantial parameter reduction while maintaining
  meaningful classification performance.

• Reproduced Fashion-MNIST QCNN performance exceeded the original
  reported accuracy under the current software environment.
""")