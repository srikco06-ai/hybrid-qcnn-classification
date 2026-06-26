"""
Prediction utilities.

Runs inference and returns:

- predicted class
- confidence score
- class probabilities
"""

import torch

from .labels import get_labels


def predict(
    model,
    image,
    dataset="mnist",
    device="cpu",
):
    """
    Perform model inference.

    Parameters
    ----------
    model
        Loaded PyTorch model.

    image
        Preprocessed image tensor.

    dataset
        "mnist" or "fashion"

    device
        cpu or cuda

    Returns
    -------
    dict
    """

    model.eval()

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

        confidence, prediction = torch.max(
            probabilities,
            dim=0,
        )

    labels = get_labels(dataset)

    prediction = int(prediction.item())

    return {
        "prediction": prediction,
        "label": labels[prediction],
        "confidence": round(
            float(confidence.item()) * 100,
            2,
        ),
        "probabilities": probabilities.cpu().tolist(),
    }