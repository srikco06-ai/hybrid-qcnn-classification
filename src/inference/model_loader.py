"""
Reusable model loader.

Loads pretrained ANN, CNN, and QCNN checkpoints.

Features
--------
- Automatic model selection
- Checkpoint validation
- Device-aware loading
- In-memory model caching
- Production-ready API
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.classical.ann_baseline import ANN
from src.classical.cnn_baseline import CNN
from src.quantum.qcnn import QCNN


# --------------------------------------------------
# Checkpoint directory
# --------------------------------------------------

CHECKPOINT_DIR = Path("src/models/checkpoints")


# --------------------------------------------------
# Available checkpoints
# --------------------------------------------------

CHECKPOINTS = {
    ("ann", "mnist"): CHECKPOINT_DIR / "ann_mnist.pth",
    ("ann", "fashion"): CHECKPOINT_DIR / "ann_fashion.pth",
    ("cnn", "mnist"): CHECKPOINT_DIR / "cnn_mnist.pth",
    ("cnn", "fashion"): CHECKPOINT_DIR / "cnn_fashion.pth",
    ("qcnn", "mnist"): CHECKPOINT_DIR / "qcnn_mnist.pth",
    ("qcnn", "fashion"): CHECKPOINT_DIR / "qcnn_fashion.pth",
}


# --------------------------------------------------
# In-memory cache
# --------------------------------------------------

_MODEL_CACHE: Dict[Tuple[str, str, str], nn.Module] = {}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def _normalize_dataset(dataset: str) -> str:
    """
    Normalize dataset names.

    Examples
    --------
    MNIST -> mnist
    Fashion-MNIST -> fashion
    fashion_mnist -> fashion
    """

    dataset = dataset.lower().strip()

    if dataset in (
        "fashion",
        "fashion-mnist",
        "fashion_mnist",
    ):
        return "fashion"

    return "mnist"


def _create_model(model_name: str) -> nn.Module:
    """
    Instantiate a model.
    """

    model_name = model_name.lower().strip()

    if model_name == "ann":
        return ANN()

    if model_name == "cnn":
        return CNN()

    if model_name == "qcnn":
        return QCNN()

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        "Choose from: ANN, CNN, QCNN."
    )


# --------------------------------------------------
# Public API
# --------------------------------------------------

def load_model(
    model_name: str,
    dataset: str,
    device: str = "cpu",
) -> nn.Module:
    """
    Load a pretrained model.

    Models are cached after the first load.

    Parameters
    ----------
    model_name
        ANN, CNN or QCNN

    dataset
        MNIST or Fashion-MNIST

    device
        cpu or cuda

    Returns
    -------
    torch.nn.Module
    """

    model_name = model_name.lower().strip()
    dataset = _normalize_dataset(dataset)

    cache_key = (
        model_name,
        dataset,
        device,
    )

    # ------------------------------------------
    # Return cached model
    # ------------------------------------------

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # ------------------------------------------
    # Build model
    # ------------------------------------------

    model = _create_model(model_name)

    checkpoint = CHECKPOINTS.get(
        (model_name, dataset)
    )

    if checkpoint is None:
        raise ValueError(
            f"No checkpoint configured for "
            f"{model_name}/{dataset}"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    # ------------------------------------------
    # Load checkpoint
    # ------------------------------------------

    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    # ------------------------------------------
    # Cache model
    # ------------------------------------------

    _MODEL_CACHE[cache_key] = model

    return model


# --------------------------------------------------
# Utility
# --------------------------------------------------

def clear_model_cache() -> None:
    """
    Clear the in-memory model cache.
    """

    _MODEL_CACHE.clear()