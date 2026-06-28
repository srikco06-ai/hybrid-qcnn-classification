from pathlib import Path

import torchvision

OUTPUT_DIR = Path("sample_images/fashion_mnist")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
)

saved_labels = set()

LABEL_NAMES = {
    0: "tshirt_top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle_boot",
}

for image, label in dataset:

    if label in saved_labels:
        continue

    filename = (
        OUTPUT_DIR
        / f"fashion_{label}_{LABEL_NAMES[label]}.png"
    )

    image.save(filename)

    print(f"Saved {filename}")

    saved_labels.add(label)

    if len(saved_labels) == 10:
        break

print(f"\nSaved {len(saved_labels)} Fashion-MNIST sample images.")