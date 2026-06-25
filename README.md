# ⚛️ Hybrid Quantum Convolutional Neural Network (QCNN) for Image Classification

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge&logo=streamlit)](https://hybrid-qcnn-classification-tiakml9nkoflebb7uthkub.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple?style=for-the-badge)

---

## Live Demo

### Interactive Streamlit Dashboard

Launch the deployed application:

<https://hybrid-qcnn-classification-tiakml9nkoflebb7uthkub.streamlit.app/>

---

## Overview

This project explores Hybrid Quantum-Classical Machine Learning using Quantum Convolutional Neural Networks (QCNNs) for image classification.

The work was originally developed as part of the IIT Delhi Continuing Education Programme in Quantum Computing and Machine Learning and later extended into a reproducible research and portfolio project.

The objective is to investigate whether parameterized quantum circuits can perform meaningful image classification while using significantly fewer trainable parameters than traditional deep learning architectures.

---

## Repository Resources

| Resource | Link |
| :--- | :--- |
| 🌐 Live Demo | <https://hybrid-qcnn-classification-tiakml9nkoflebb7uthkub.streamlit.app/> |
| 💻 GitHub Repository | <https://github.com/srikco06-ai/hybrid-qcnn-classification> |
| 📄 Project Report | `docs/QCNN_Report_V1.0.pdf` |
| 📊 Presentation | `docs/GROUP-8_QCNN_MINST_FashionMNIST.pptx` |

---

## Key Highlights

| Metric | Value |
| :--- | ---: |
| QCNN Parameters | 9,482 |
| CNN Parameters | 421,642 |
| ANN Parameters | 235,146 |
| MNIST QCNN Accuracy | 82.31% |
| Fashion-MNIST QCNN Accuracy | 78.44% |
| Quantum Framework | PennyLane |
| Deep Learning Framework | PyTorch |

### Main Finding

The QCNN achieved meaningful classification performance while using approximately **44× fewer trainable parameters** than the CNN baseline.

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

- Hybrid Quantum-Classical Learning Architecture
- Quantum Convolutional Neural Network (QCNN)
- Variational Quantum Circuits (VQC)
- Angle Embedding
- Quantum Convolution Layer
- Quantum Pooling Layer
- Strongly Entangling Layers
- ANN vs CNN vs QCNN Benchmarking
- MNIST Classification
- Fashion-MNIST Classification
- Interactive Streamlit Dashboard
- PennyLane + PyTorch Integration

---

## QCNN Architecture

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

- 60,000 training images
- 10,000 testing images
- 28×28 grayscale
- 10 classes

### Fashion-MNIST Dataset

- 60,000 training images
- 10,000 testing images
- 28×28 grayscale
- 10 classes

---

## Original IIT Delhi Project Results

### Original MNIST Results

| Model | Accuracy |
| :--- | ---: |
| ANN | 98.00% |
| CNN | 98.81% |
| QCNN | 82.35% |

### Original Fashion-MNIST Results

| Model | Accuracy |
| :--- | ---: |
| ANN | 88.74% |
| CNN | 92.06% |
| QCNN | 59.35% |

---

## Reproduced Repository Results

### Reproduced MNIST Results

| Model | Accuracy |
| :--- | ---: |
| ANN | 97.49% |
| CNN | 99.12% |
| QCNN | 82.31% |

### Reproduced Fashion-MNIST Results

| Model | Accuracy |
| :--- | ---: |
| ANN | 87.34% |
| CNN | 91.59% |
| QCNN | 78.44% |

---

## Parameter Comparison

| Model | Parameters |
| :--- | ---: |
| ANN | 235,146 |
| CNN | 421,642 |
| QCNN | 9,482 |

QCNN uses approximately **44× fewer parameters** than the CNN baseline.

---

## Reproduction Analysis

### Key Findings

- QCNN reproduced MNIST performance within 0.04%.
- Fashion-MNIST performance improved significantly.
- CNN achieved the highest overall accuracy.
- QCNN demonstrated strong parameter efficiency.

### Reproduction Notes

The implementation was reproduced independently using PennyLane and PyTorch. Small differences may arise from software versions, initialization, optimization, and hardware.

---

## Technology Stack

### Quantum Computing

- PennyLane
- Variational Quantum Circuits
- QCNN

### Machine Learning

- PyTorch
- NumPy
- Pandas
- Scikit-Learn

### Visualization

- Streamlit
- Plotly
- Matplotlib

---

## Project Structure

```text
hybrid-qcnn-classification/
├── app/
├── docs/
├── screenshots/
├── src/
├── LICENSE
├── README.md
└── requirements.txt
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

## Run

```bash
streamlit run app/Home.py
```

---

## Future Improvements

- Quantum Vision Transformer
- Larger datasets
- Real quantum hardware
- Live inference
- Additional quantum architectures

---

## Author

### Sri Krishna Chaitanya Ogirala

AI & Machine Learning Engineer

- Quantum Computing
- FastAPI
- Python
- Next.js
- Generative AI

---

## License

Licensed under the MIT License.
