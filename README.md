# 🧠 FocusedKAN: Adaptive Attention with Kolmogorov–Arnold Networks for Multi-label Mental Health Prediction

**FocusedKAN** is a custom deep learning architecture designed for accurate and interpretable mental health condition classification from Reddit text data. It combines the expressiveness of Kolmogorov–Arnold Networks (KANs) with transformer-based architectures and a novel attention-head freezing strategy to enable efficient and effective multi-label learning.

---

## 📘 Project Overview

The model classifies text into five mental health conditions:

- **Depression**
- **Anxiety**
- **OCD**
- **PTSD**
- **Bipolar Disorder**

It is trained using a **two-stage learning strategy**:

1. **Binary classification phase** with per-disorder training using independent Transformer heads and adaptive attention freezing.
2. **Multi-label fine-tuning phase** that jointly predicts all five disorders using shared representations.

---

## 🔧 Core Innovations

### 🔹 Kolmogorov–Arnold Networks (KANs)
- Replaces standard MLP feedforward layers in Transformers with **KANLinear layers** using learnable spline activations for greater expressiveness.
- Implemented custom `KANLinear` module with B-spline interpolation and per-neuron activation learning.

### 🔹 Adaptive Freezing of Attention Heads
- Introduced the **“Refrigerator”** module to dynamically monitor and freeze attention heads based on stability of their weights.
- Prevents overfitting and encourages parameter reuse across tasks.

### 🔹 Stage-wise Knowledge Transfer
- Employs **curriculum-style training**, where each binary disorder model is trained in sequence with head freezing.
- Final multi-label head learns from the enriched representation learned during binary training.

---

## 📊 Dataset

- Source: Reddit mental health subreddits and control groups
- 60,000 training samples per class for binary classification (5 binary datasets)
- A combined multi-label dataset of 60,000 samples used for Stage 2

---

## 🧱 Architecture Overview

- Base encoder: **BERT (uncased)** from Hugging Face
- Transformer block with custom attention + KAN-enhanced FFN layers
- Separate binary heads for Stage 1, combined head for Stage 2
- Fully integrated with Hugging Face `Trainer`

