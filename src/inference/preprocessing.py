"""
Image preprocessing utilities.

Converts uploaded images into tensors compatible with
ANN, CNN, and QCNN models.
"""

from PIL import Image

import torch
from torchvision import transforms


MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,),
        ),
    ]
)

FASHION_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,),
        ),
    ]
)


def preprocess_image(image, dataset="mnist"):
    """
    Convert uploaded image into model input tensor.

    Parameters
    ----------
    image : PIL.Image

    dataset : str
        mnist
        fashion

    Returns
    -------
    torch.Tensor
        Shape:
            (1,1,28,28)
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    dataset = dataset.lower()

    if dataset == "mnist":
        tensor = MNIST_TRANSFORM(image)

    elif dataset in (
        "fashion",
        "fashion-mnist",
        "fashion_mnist",
    ):
        tensor = FASHION_TRANSFORM(image)

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    return tensor.unsqueeze(0)