# Spatial-Temporal Synergy in Hybrid CNN-RNN and Vision Transformers

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-Supported-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey)
![Accuracy](https://img.shields.io/badge/Hybrid_Model_Accuracy-92.24%25-success)

> **Official codebase for the research paper:** *[A Novel Approach for Evaluating Spatial-Temporal Synergy in Hybrid CNN-RNN and Vision Transformer Architectures](https://www.igi-global.com/gateway/article/full-text-html/411189).*

## 📖 Overview

As visual data grows in volume and complexity, robust models for accurate pattern interpretation are essential. This repository explores and evaluates advanced deep learning architectures for multi-class image recognition using the **CIFAR-10** benchmark dataset. 

We conduct a comparative analysis between traditional Convolutional Neural Networks (CNNs), modern **Vision Transformers (ViTs)**, and a highly optimized **Hybrid CNN-RNN architecture**. By fusing spatial feature extraction (CNN) with temporal sequence modeling (RNN/LSTM), our hybrid model successfully captures rich, spatiotemporal dependencies, achieving a superior test accuracy of **92.24%**.

---

## ✨ Key Research Contributions

* **Hybrid Architecture Implementation:** Integrated CNNs (for spatial edge/texture extraction) and LSTMs (for temporal relationship modeling) to process visual data sequentially.
* **Comparative Evaluation:** Benchmarked traditional CNNs, simple RNNs, Vision Transformers, and the proposed Hybrid model in a unified experimental framework.
* **Spatial-Temporal Synergy:** Demonstrated how integrating convolutional mapping with sequence processing mitigates information loss and improves generalization.
* **High-Fidelity Classification:** Achieved precision and recall scores exceeding 90% across CIFAR-10 classes, outperforming standalone baselines significantly.

---

## 📊 Performance & Results

The models were trained over 100 epochs using dynamic learning rate scheduling and optimized via Rectified Adam. The Hybrid CNN-RNN demonstrated the highest generalization capabilities.

| Model Architecture | Training Accuracy | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | ~88.00% | ~68.00% | 0.67 | 0.66 | 0.66 |
| **Simple RNN** | ~80.00% | 62.50% | 0.60 | 0.61 | 0.60 |
| **Vision Transformer (ViT)** | ~90.00% | ~84.00% | 0.83 | 0.84 | 0.83 |
| **🏆 Hybrid CNN-RNN** | **95.90%** | **92.24%** | **0.92** | **0.92** | **0.92** |

*(Note: The `confusion_matrix.png` included in this repository provides a granular, class-wise breakdown of the Hybrid model's predictions).*

---

## 📂 Repository Structure

```text
Hybrid_Pattern_Recognition/
│
├── models/
│   ├── hybrid_cnn_rnn.py          # Core architecture for the Hybrid CNN-LSTM model
│   └── vision_transformer.py      # Implementation of the ViT baseline
│
├── main.py                        # Entry point for training and evaluation
├── requirements.txt               # Python dependencies
├── confusion_matrix.png           # Visualized evaluation results
│
├── hybrid_cnn_rnn.h5              # Standard saved model weights
├── hybrid_cnn_rnn_best.h5         # Best performing model weights (Early Stopping)
└── hybrid_cnn_rnn_improved.keras  # Final optimized model in Keras format
```

## 🚀 Quick Start
1. Clone the repository
Bash
git clone [https://github.com/VirenPassi/Hybrid_Pattern_Recognition.git](https://github.com/VirenPassi/Hybrid_Pattern_Recognition.git)
cd Hybrid_Pattern_Recognition
2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:

Bash
pip install -r requirements.txt
3. Run the Evaluation
You can load the pre-trained weights (.keras or .h5 files) and evaluate the model on the CIFAR-10 dataset using the main execution script:

Bash
python main.py

## 🎓 Citation
If you find this code or research helpful in your work, please consider citing the original paper published in IGI Global. Read the full article here.

Code snippet
@article{passi2026spatialtemporal,
  title={A Novel Approach for Evaluating Spatial-Temporal Synergy in Hybrid CNN-RNN and Vision Transformer Architectures},
  author={Passi, Viren and Kumar, Sudhakar and Singh, Sunil K. and Verma, Shreya and Arya, Varsha and Tang, Valerie and Gupta, Brij B. and Chui, Kwok Tai},
  journal={International Journal of Information Technology (IJIIT)},
  year={2026},
  publisher={IGI Global},
  url={[https://www.igi-global.com/gateway/article/full-text-html/411189](https://www.igi-global.com/gateway/article/full-text-html/411189)}
}

## Authors & Affiliations:

Viren Passi, Chandigarh College of Engineering and Technology, Chandigarh, India

Shreya Verma, Chandigarh College of Engineering and Technology, Chandigarh, India

(See full publication for the complete author list and affiliations).
