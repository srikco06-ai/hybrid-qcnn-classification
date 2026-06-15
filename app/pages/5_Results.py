import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Results")

mnist = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [98.00, 98.81, 82.35]
})

fashion = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Accuracy": [88.74, 92.06, 59.35]
})

tab1, tab2 = st.tabs(
    ["MNIST", "Fashion-MNIST"]
)

with tab1:

    fig = px.bar(
        mnist,
        x="Model",
        y="Accuracy",
        title="MNIST Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    fig = px.bar(
        fashion,
        x="Model",
        y="Accuracy",
        title="Fashion-MNIST Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

parameters = pd.DataFrame({
    "Model": ["ANN", "CNN", "QCNN"],
    "Parameters": [235146, 421642, 9482]
})

fig = px.bar(
    parameters,
    x="Model",
    y="Parameters",
    title="Parameter Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.success(
    "QCNN achieved approximately 44× parameter reduction compared with CNN."
)