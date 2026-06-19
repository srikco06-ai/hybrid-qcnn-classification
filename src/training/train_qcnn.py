import json
from pathlib import Path

import torch
import torch.nn as nn

from src.data.datasets import get_mnist_loaders
from src.quantum.qcnn import QCNN
from src.utils import (
    get_device,
    count_parameters
)

EPOCHS = 10
LEARNING_RATE = 0.002

CHECKPOINT_PATH = (
    "src/models/checkpoints/qcnn_mnist.pth"
)

METRICS_PATH = (
    "src/models/metrics/qcnn_results.json"
)


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


def train():

    device = get_device()

    print(f"Using device: {device}")

    train_loader, test_loader = (
        get_mnist_loaders()
    )

    model = QCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        accuracy = evaluate(
            model,
            test_loader,
            device
        )

        best_accuracy = max(
            best_accuracy,
            accuracy
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    Path(
        "src/models/checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        "src/models/metrics"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH
    )

    metrics = {
        "dataset": "MNIST",
        "model": "QCNN",
        "accuracy": round(
            best_accuracy,
            2
        ),
        "epochs": EPOCHS,
        "parameters": count_parameters(
            model
        )
    }

    with open(
        METRICS_PATH,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"\nCheckpoint saved: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Metrics saved: "
        f"{METRICS_PATH}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )


if __name__ == "__main__":
    train()