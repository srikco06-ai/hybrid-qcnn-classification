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
Measurement Layer
      │
      ▼
Classification Output
```

---

## Datasets

### MNIST Dataset

* Handwritten digit recognition
* 28 × 28 grayscale images
* 10 classification classes
* Standard computer vision benchmark

### Fashion-MNIST Dataset

* Fashion article classification
* 28 × 28 grayscale images
* 10 classification classes
* More challenging than MNIST

---

## Model Comparison

| Model | Architecture Type                |
| ----- | -------------------------------- |
| ANN   | Fully Connected Neural Network   |
| CNN   | Convolutional Neural Network     |
| QCNN  | Hybrid Quantum-Classical Network |

---

## Experimental Results

### MNIST Results

| Model | Accuracy |
| ----- | -------- |
| ANN   | 98.00%   |
| CNN   | 98.81%   |
| QCNN  | 82.35%   |

### Fashion-MNIST Results

| Model | Accuracy |
| ----- | -------- |
| ANN   | 88.74%   |
| CNN   | 92.06%   |
| QCNN  | 59.35%   |

### Parameter Efficiency

| Model | Parameters |
| ----- | ---------- |
| ANN   | 235,146    |
| CNN   | 421,642    |
| QCNN  | 9,482      |

The QCNN achieved approximately **44× fewer trainable parameters** than the CNN baseline while maintaining meaningful classification performance.

---

## Technology Stack

### Quantum Computing

* PennyLane
* Variational Quantum Circuits
* Quantum Convolutional Neural Networks

### Machine Learning

* PyTorch
* Scikit-Learn
* NumPy
* Pandas

### Application Layer

* Streamlit
* Plotly

---

## Project Structure

```text
hybrid-qcnn-classification
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Project_Overview.py
│       ├── 2_Architecture.py
│       ├── 3_Dataset.py
│       ├── 4_Model_Comparison.py
│       └── 5_Results.py
│
├── src/
│   ├── classical/
│   │   ├── ann_baseline.py
│   │   └── cnn_baseline.py
│   │
│   └── quantum/
│       ├── circuits.py
│       ├── qcnn.py
│       └── inference.py
│
├── docs/
├── screenshots/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repository-url>

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
* Explore alternative quantum embedding strategies
* Implement Quantum Vision Transformer architectures
* Add live image inference capability
* Evaluate larger benchmark datasets
* Compare additional quantum neural network architectures
* Deploy as a public web application

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Quantum Computing Fundamentals
* Quantum Machine Learning
* Variational Quantum Circuits
* Quantum Convolutional Neural Networks
* Hybrid Quantum-Classical Architectures
* Quantum Feature Encoding
* Model Evaluation and Benchmarking

---

## Author

### Sri Krishna Chaitanya

AI & Machine Learning Engineer

Quantum Computing & Quantum Machine Learning

FastAPI • Python • Next.js • GenAI Applications
