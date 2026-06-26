"""
Reusable model loader.

Loads pretrained ANN, CNN, and QCNN checkpoints.
"""

from pathlib import Path

import torch

from src.classical.ann_baseline import ANN
from src.classical.cnn_baseline import CNN
from src.quantum.qcnn import QCNN


CHECKPOINT_DIR = Path(
    "src/models/checkpoints"
)


CHECKPOINTS = {
    ("ann", "mnist"):
        CHECKPOINT_DIR / "ann_mnist.pth",

    ("ann", "fashion"):
        CHECKPOINT_DIR / "ann_fashion.pth",

    ("cnn", "mnist"):
        CHECKPOINT_DIR / "cnn_mnist.pth",

    ("cnn", "fashion"):
        CHECKPOINT_DIR / "cnn_fashion.pth",

    ("qcnn", "mnist"):
        CHECKPOINT_DIR / "qcnn_mnist.pth",

    ("qcnn", "fashion"):
        CHECKPOINT_DIR / "qcnn_fashion.pth",
}


def load_model(
    model_name: str,
    dataset: str,
    device="cpu",
):
    """
    Load pretrained model.

    Parameters
    ----------
    model_name
        ann
        cnn
        qcnn

    dataset
        mnist
        fashion
    """

    model_name = model_name.lower()
    dataset = dataset.lower()

    if dataset in (
        "fashion-mnist",
        "fashion_mnist",
    ):
        dataset = "fashion"

    if model_name == "ann":
        model = ANN()

    elif model_name == "cnn":
        model = CNN()

    elif model_name == "qcnn":
        model = QCNN()

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    checkpoint = CHECKPOINTS.get(
        (
            model_name,
            dataset,
        )
    )

    if checkpoint is None:
        raise ValueError(
            "Checkpoint not configured."
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model