"""
High-level inference engine for the Hybrid QCNN Classification project.

This module provides a reusable interface for
running end-to-end inference.

Used by

- Streamlit
- FastAPI
- Unit Tests
- Future API clients
"""

from time import perf_counter

from .model_loader import load_model
from .predictor import predict
from .preprocessing import preprocess_image


class InferenceEngine:
    """
    End-to-end inference engine.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def predict_image(
        self,
        image,
        model_name,
        dataset,
    ):
        """
        Run complete inference pipeline.

        Parameters
        ----------
        image
            Uploaded PIL image

        model_name
            ANN
            CNN
            QCNN

        dataset
            MNIST
            Fashion-MNIST

        Returns
        -------
        dict
        """

        start = perf_counter()

        tensor = preprocess_image(
            image=image,
            dataset=dataset,
        )

        model = load_model(
            model_name=model_name,
            dataset=dataset,
            device=self.device,
        )

        result = predict(
            model=model,
            image=tensor,
            dataset=dataset,
            device=self.device,
        )

        elapsed = (
            perf_counter() - start
        ) * 1000

        result["model"] = model_name.upper()

        result["dataset"] = dataset

        result["inference_time"] = round(
            elapsed,
            2,
        )

        return result