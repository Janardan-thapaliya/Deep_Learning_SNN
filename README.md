# Deep Learning — Spiking Neural Networks (SNN) & Neuromorphic Vision

Hands-on notebooks exploring Spiking Neural Networks (SNNs) and hybrid SNN-CNN architectures trained on neuromorphic event-camera datasets, including N-MNIST, N-Caltech101, and custom event-frame data.

---

## Repository Structure

```
Deep_Learning_SNN/
├── N_MNIST_SNN.ipynb                    ← SNN trained on neuromorphic N-MNIST
├── N_Caltech101_HybridSNN_CNN.ipynb     ← Hybrid SNN-CNN on N-Caltech101
├── SNN_Caltech101.ipynb                 ← Pure SNN on Caltech101 event data
├── Object_detection_v1.ipynb            ← Object detection on event-camera frames
└── Event_Camera_data_Superframes.ipynb  ← Event stream → superframe preprocessing
```

---

## Notebooks

### 1. `N_MNIST_SNN.ipynb`
Spiking Neural Network trained on N-MNIST — a neuromorphic version of the MNIST handwritten digit dataset captured using a Dynamic Vision Sensor (DVS).

- **Dataset:** N-MNIST (70,000 event-based samples, 10 digit classes)
- **Architecture:** Leaky Integrate-and-Fire (LIF) spiking neurons with spike-based forward pass
- **Key concepts:** Event-to-frame conversion, spike encoding, spiking neuron dynamics

---

### 2. `N_Caltech101_HybridSNN_CNN.ipynb`
Hybrid architecture combining spiking and conventional convolutional layers for multi-class classification on N-Caltech101.

- **Dataset:** N-Caltech101 (neuromorphic version of Caltech101, 101 object categories)
- **Architecture:** SNN feature extractor + CNN classification head
- **Key concepts:** Hybrid SNN-CNN design, neuromorphic multi-class classification

---

### 3. `SNN_Caltech101.ipynb`
Pure SNN pipeline applied to Caltech101 event data for object category recognition.

- **Dataset:** N-Caltech101
- **Architecture:** Multi-layer SNN with LIF neurons
- **Key concepts:** Deep spiking networks, bio-inspired computing, temporal spike coding

---

### 4. `Object_detection_v1.ipynb`
Object detection pipeline built on top of event-camera frame representations.

- **Approach:** Frame-based detection on converted event superframes
- **Key concepts:** Bounding box prediction, event-to-frame preprocessing for detection tasks

---

### 5. `Event_Camera_data_Superframes.ipynb`
Preprocessing pipeline that converts raw event-camera streams into superframe representations suitable for CNN or SNN input.

- **Key concepts:** DVS event accumulation, temporal binning, superframe construction, normalization

---

## Tech Stack

SnnTorch · PyTorch · NumPy · Matplotlib · Python

---

## Setup

```bash
pip install snntorch torch torchvision numpy matplotlib
```
