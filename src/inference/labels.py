"""
Dataset label definitions.
"""

MNIST_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

FASHION_MNIST_LABELS = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def get_labels(dataset: str):
    """
    Return label mapping for a dataset.

    Parameters
    ----------
    dataset : str
        MNIST or Fashion-MNIST

    Returns
    -------
    dict
    """

    dataset = dataset.lower()

    if dataset == "mnist":
        return MNIST_LABELS

    if dataset in ["fashion", "fashion-mnist", "fashion_mnist"]:
        return FASHION_MNIST_LABELS

    raise ValueError(f"Unsupported dataset: {dataset}")