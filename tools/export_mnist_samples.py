from pathlib import Path

from torchvision.datasets import MNIST

OUTPUT_DIR = Path("sample_images/mnist")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = MNIST(
    root="./data",
    train=False,
    download=True,
)

saved_labels = set()

for image, label in dataset:

    if label in saved_labels:
        continue

    filename = OUTPUT_DIR / f"mnist_{label}.png"

    image.save(filename)

    print(f"Saved {filename}")

    saved_labels.add(label)

    if len(saved_labels) == 10:
        break

print(f"\nSaved {len(saved_labels)} MNIST sample images.")