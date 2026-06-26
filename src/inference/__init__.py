"""
Inference package for Hybrid QCNN Classification.

This package provides reusable utilities for:

- Image preprocessing
- Model loading
- Prediction
- Dataset labels

These modules are shared by:

- Streamlit Dashboard
- FastAPI
- Unit Tests
- Future API clients
"""
from .inference_engine import InferenceEngine

from .labels import (
    MNIST_LABELS,
    FASHION_MNIST_LABELS,
    get_labels,
)

from .preprocessing import preprocess_image

from .model_loader import load_model

from .predictor import predict

__all__ = [
    "MNIST_LABELS",
    "FASHION_MNIST_LABELS",
    "get_labels",
    "preprocess_image",
    "load_model",
    "predict",
    "InferenceEngine",
]