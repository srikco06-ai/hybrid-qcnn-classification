# ⚛️ Hybrid Quantum Convolutional Neural Network (QCNN) for Image Classification

## Overview

This project explores Hybrid Quantum-Classical Machine Learning using Quantum Convolutional Neural Networks (QCNNs) for image classification.

The objective is to investigate whether parameterized quantum circuits can perform meaningful feature extraction while using significantly fewer trainable parameters than traditional deep learning architectures.

Developed as part of the IIT Delhi Continuing Education Programme in Quantum Computing and Machine Learning.

---

## Key Features

* Hybrid Quantum-Classical Learning Architecture
* Quantum Convolutional Neural Network (QCNN)
* Variational Quantum Circuits (VQC)
* Angle Embedding for Quantum Feature Encoding
* Comparative Analysis against ANN and CNN Baselines
* MNIST and Fashion-MNIST Benchmark Evaluation
* Interactive Streamlit Dashboard
* PennyLane + PyTorch Integration

---

## Research Objectives

This project investigates:

* Quantum feature extraction using variational quantum circuits
* Parameter-efficient learning in near-term quantum systems
* Performance comparison between classical and quantum models
* Practical applications of Quantum Machine Learning for image classification

---

## Architecture

### Hybrid QCNN Pipeline

```text
Input Image
      │
      ▼
Feature Encoding
      │
      ▼
Angle Embedding
      │
      ▼
Quantum Convolution Layer
      │
      ▼
Quantum Pooling Layer
      │
      ▼
Strongly Entangling Layers
      │
      ▼
Measurement Layer
      │
      ▼
Classification Output
```

---

## Datasets

### MNIST Dataset

* Handwritten digit recognition
* 28×28 grayscale images
* 10 classification classes

### Fashion-MNIST Dataset

* Fashion article classification
* 28×28 grayscale images
* 10 classification classes

---

## Experimental Results

### Original IIT Delhi Project Results

#### MNIST

| Model | Accuracy |
| ----- | -------: |
| ANN   |   98.00% |
| CNN   |   98.81% |
| QCNN  |   82.35% |

#### Fashion-MNIST

| Model | Accuracy |
| ----- | -------: |
| ANN   |   88.74% |
| CNN   |   92.06% |
| QCNN  |   59.35% |

---

### Reproduced Results

#### MNIST

| Model | Accuracy |
| ----- | -------: |
| ANN   |   97.49% |
| CNN   |   99.12% |
| QCNN  |   82.31% |

#### Fashion-MNIST

| Model | Accuracy |
| ----- | -------: |
| ANN   |   87.34% |
| CNN   |   91.59% |
| QCNN  |   78.44% |

---

## Parameter Efficiency

| Model | Parameters |
| ----- | ---------: |
| ANN   |    235,146 |
| CNN   |    421,642 |
| QCNN  |      9,482 |

QCNN achieves substantial parameter reduction while maintaining meaningful classification performance.

---

## Technology Stack

### Quantum Computing

* PennyLane
* Variational Quantum Circuits
* Quantum Convolutional Neural Networks

### Machine Learning

* PyTorch
* NumPy
* Pandas
* Scikit-Learn

### Application Layer

* Streamlit
* Plotly
* Matplotlib

---

## Setup

```bash
git clone https://github.com/srikco06-ai/hybrid-qcnn-classification.git

cd hybrid-qcnn-classification

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Launch Dashboard

```bash
streamlit run app/Home.py
```

---

## Future Improvements

* Increase quantum circuit depth
* Explore alternative quantum embeddings
* Implement Quantum Vision Transformers
* Add image inference UI
* Evaluate larger datasets
* Benchmark on quantum hardware

---

## Author

### Sri Krishna Chaitanya

AI & Machine Learning Engineer

Quantum Computing & Quantum Machine Learning

FastAPI • Python • Next.js • GenAI Applications

---

## License

MIT License
