# ⚛️ Hybrid Quantum Convolutional Neural Network (QCNN) for Image Classification

## Overview

This project explores Hybrid Quantum-Classical Machine Learning using Quantum Convolutional Neural Networks (QCNNs) for image classification.

The work was originally developed as part of the IIT Delhi Continuing Education Programme in Quantum Computing and Machine Learning and later extended into a reproducible research and portfolio project.

The objective is to investigate whether parameterized quantum circuits can perform meaningful image classification while using significantly fewer trainable parameters than traditional deep learning architectures.

---

## Key Highlights

| Metric                      | Value     |
| --------------------------- | --------- |
| QCNN Parameters             | 9,482     |
| CNN Parameters              | 421,642   |
| ANN Parameters              | 235,146   |
| MNIST QCNN Accuracy         | 82.31%    |
| Fashion-MNIST QCNN Accuracy | 78.44%    |
| Quantum Framework           | PennyLane |
| Deep Learning Framework     | PyTorch   |

### Main Finding

The QCNN achieved meaningful classification performance while using approximately **44× fewer trainable parameters** than the CNN baseline.

---

## Project Documentation

This repository includes the original project deliverables:

* `docs/QCNN_Report_V1.0.pdf`
* `docs/GROUP-8_QCNN_MINST_FashionMNIST.pptx`

---

## Screenshots

### Home Dashboard

![Home](screenshots/home.png)

### Architecture Dashboard

![Architecture](screenshots/architecture.png)

### Model Comparison

![Comparison](screenshots/comparison.png)

### Experimental Results Dashboard

![Results](screenshots/results.png)

### QCNN Implementation Dashboard

![QCNN Implementation](screenshots/qcnn_implementation.png)

---

## Features

* Hybrid Quantum-Classical Learning Architecture
* Quantum Convolutional Neural Network (QCNN)
* Variational Quantum Circuits (VQC)
* Angle Embedding
* Quantum Convolution
* Quantum Pooling
* Strongly Entangling Layers
* ANN vs CNN vs QCNN Benchmarking
* MNIST Classification
* Fashion-MNIST Classification
* Interactive Streamlit Dashboard
* PennyLane + PyTorch Integration

---

## QCNN Architecture

### Hybrid Quantum Pipeline

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
* 60,000 training samples
* 10,000 testing samples
* 10 classes

### Fashion-MNIST Dataset

* Fashion article classification
* 28×28 grayscale images
* 60,000 training samples
* 10,000 testing samples
* 10 classes

---

## Original IIT Delhi Project Results

### Original MNIST Results

| Model | Accuracy |
| ----- | -------: |
| ANN   |   98.00% |
| CNN   |   98.81% |
| QCNN  |   82.35% |

### Original Fashion-MNIST Results

| Model | Accuracy |
| ----- | -------: |
| ANN   |   88.74% |
| CNN   |   92.06% |
| QCNN  |   59.35% |

---

## Reproduced Repository Results

### Reproduced MNIST Results

| Model | Accuracy |
| ----- | -------: |
| ANN   |   97.49% |
| CNN   |   99.12% |
| QCNN  |   82.31% |

### Reproduced Fashion-MNIST Results

| Model | Accuracy |
| ----- | -------: |
| ANN   |   87.34% |
| CNN   |   91.59% |
| QCNN  |   78.44% |

---

## Parameter Comparison

| Model | Parameters |
| ----- | ---------: |
| ANN   |    235,146 |
| CNN   |    421,642 |
| QCNN  |      9,482 |

QCNN uses approximately **44× fewer parameters** than the CNN baseline.

---

## Reproduction Analysis

The reproduced implementation successfully validated the original QCNN architecture.

### Key Findings

* QCNN reproduced MNIST performance within 0.04% of the original project.
* Fashion-MNIST performance improved significantly under the current software environment.
* CNN remained the highest-performing architecture on both datasets.
* QCNN demonstrated strong parameter efficiency with only 9,482 trainable parameters.
* The project validates the feasibility of Hybrid Quantum-Classical Learning for image classification tasks.

### Reproduction Notes

The reproduced implementation was developed independently using PennyLane and PyTorch based on the original IIT Delhi QCNN project framework.

Minor differences between the original and reproduced results may arise from variations in software versions, random initialization, optimization behavior, training environments, and hardware configurations.

The primary objective of the reproduction study was to validate the QCNN architecture, benchmark its parameter efficiency, and evaluate its performance on MNIST and Fashion-MNIST datasets using a standalone implementation.

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

### Visualization and Dashboard

* Streamlit
* Plotly
* Matplotlib

---

## Project Structure

```text
hybrid-qcnn-classification
│
├── app/
│   ├── Home.py
│   └── pages/
│
├── src/
│   ├── classical/
│   ├── quantum/
│   ├── training/
│   └── data/
│
├── screenshots/
├── docs/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/srikco06-ai/hybrid-qcnn-classification.git

cd hybrid-qcnn-classification

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app/Home.py
```

---

## Live Demo

Streamlit Application:

Coming Soon

> The application will be deployed on Streamlit Community Cloud.

---

## Future Improvements

* Increase quantum circuit depth
* Explore alternative quantum embedding strategies
* Implement Quantum Vision Transformer architectures
* Add image upload and live inference
* Evaluate larger benchmark datasets
* Benchmark on real quantum hardware
* Compare additional quantum neural network architectures

---

## Author

### Sri Krishna Chaitanya

AI & Machine Learning Engineer

Quantum Computing & Quantum Machine Learning

FastAPI • Python • Next.js • GenAI Applications

---

## License

MIT License
