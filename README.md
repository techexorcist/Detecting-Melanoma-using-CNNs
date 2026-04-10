# 🔬 Detecting Melanoma using CNNs: A Deep Learning Approach

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-D00000.svg)](https://keras.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23.5-013243.svg)](https://numpy.org/)

> **Author:** Vinodh Nagarajaiah  
> **Programme:** AI/ML Executive Programme (UpGrad & IIIT-B)

## ⏱️ Executive Summary (TL;DR)
* **The Goal:** Build a custom Convolutional Neural Network (CNN) to accurately classify and detect melanoma and 8 other oncological skin diseases, reducing the manual diagnostic effort required by dermatologists.
* **The Data:** Analysed a highly imbalanced dataset of 2,357 images from the International Skin Imaging Collaboration (ISIC).
* **The Process:** Executed a complete deep learning pipeline—from baseline CNN architecture to advanced regularisation (Dropout, Data Augmentation) and systematic class-imbalance resolution using the `Augmentor` library.
* **The Result:** Successfully identified model overfitting in baseline architectures, subsequently stabilising validation accuracy and improving generalisation by dynamically synthesising 500 new training samples per class. The final model achieved a highly robust **Validation Accuracy of 82.35%**.

---

## 📖 Table of Contents
1. [Problem Statement & Objective](#-problem-statement--objective)
2. [Skills & Deep Learning Competencies](#-skills--deep-learning-competencies)
3. [Methodology: The Modelling Pipeline](#-methodology-the-modelling-pipeline)
4. [Key Insights & Model Evaluation](#-key-insights--model-evaluation)
5. [Strategic Clinical Recommendations](#-strategic-clinical-recommendations)
6. [Future Scope & Improvements](#-future-scope--improvements)
7. [Repository Structure](#-repository-structure)
8. [Acknowledgements & Contact](#-acknowledgements--contact)

---

## 💼 Problem Statement & Objective
Melanoma is a severe type of skin cancer that can be deadly if not detected early. It accounts for a staggering **75% of skin cancer deaths**. 

**The Core Objective:** To develop an automated, CNN-based image classification model capable of evaluating skin lesions and alerting medical professionals to the presence of melanoma. By accurately distinguishing between 9 specific oncological diseases, this model aims to serve as an early-warning diagnostic tool, improving patient outcomes and streamlining clinical workflows.

---

## 🛠️ Skills & Deep Learning Competencies
* **Computer Vision:** Building multi-layer Convolutional Neural Networks (CNNs) using `TensorFlow` and `Keras`.
* **Image Preprocessing:** Normalising pixel ranges (0-1), scaling image resolution ($180 \times 180$), and implementing batched dataset loading via `image_dataset_from_directory`.
* **Model Regularisation:** Applying `Dropout` layers and on-the-fly Data Augmentation (random flips, rotations, zooms) to combat network overfitting.
* **Class Imbalance Handling:** Utilising the `Augmentor` pipeline to synthetically generate robust, balanced class distributions.
* **Performance Diagnostics:** Plotting epoch history to mathematically diagnose underfitting vs. overfitting scenarios.

---

## 🧠 Methodology: The Modelling Pipeline

### 1. Data Ingestion & Understanding
* **The Dataset:** [Download the raw dataset here](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing). 
* Sourced 2,357 images across 9 skin disease categories (e.g., Actinic keratosis, Basal cell carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented benign keratosis, Seborrheic keratosis, Squamous cell carcinoma, and Vascular lesion). 
* These images were formed from the **International Skin Imaging Collaboration (ISIC)**. 
* Configured a scalable training/validation split (80/20) to ensure isolated model evaluation (1792 training images, 447 validation).

### 2. Baseline Model Building (Vanilla CNN)
* Constructed a standard sequential CNN with repeated `Conv2D` and `MaxPooling2D` layers.
* **Result:** The model severely overfit the training data (memorising noise rather than generalising). 

### 3. Implementing Regularisation
* Rebuilt the architecture to include spatial **Data Augmentation** layers to artificially diversify the training images.
* Integrated **Dropout (0.2)** layers to force the network to rely on a broader array of learned features.
* **Result:** Overfitting was drastically reduced, but the model struggled to increase overall accuracy, revealing a deeper data issue.

### 4. Addressing Class Imbalance
* A diagnostic check revealed a severe class imbalance: *Seborrheic keratosis* had only **77 samples**, while *Pigmented benign keratosis* dominated with **462 samples**.
* Deployed the `Augmentor` library to synthetically generate **500 new samples** for every single class, establishing a perfectly balanced data environment for the final network iteration.

---

## 📊 Key Insights & Model Evaluation

### Model 1: The Overfitting Baseline
* **Train Accuracy:** ~89.29%
* **Validation Accuracy:** ~54.81%
* **Analysis:** The massive gap between train and validation scores indicated severe overfitting. The network memorised the training images rather than learning generalisable features.

### Model 2: Augmented & Regularised
* **Train Accuracy:** ~82.92%
* **Validation Accuracy:** ~53.24%
* **Analysis:** The training accuracy dropped, confirming that the `Dropout` and `Data Augmentation` successfully prevented the model from simply memorising the data. However, the plateau in validation accuracy proved that the dataset was fundamentally too small and imbalanced.

### Model 3: Final Rectification (Augmentor)
* **Train Accuracy:** ~92.20%
* **Validation Accuracy:** ~82.35%
* **Analysis:** By identifying the sparse classes and using `Augmentor` to balance the dataset, the foundational roadblocks preventing high-confidence clinical predictions were systematically removed. The validation accuracy surged to 82.35%, proving the model learned genuine, disease-specific features.

---

## 💡 Strategic Clinical Recommendations
1. **Early Diagnostic Alerts:** Once deployed, the model should not replace dermatologists, but rather act as a "triage" system, flagging high-risk images (like potential Melanomas) for immediate human review.
2. **Standardised Image Capture:** The model's accuracy is heavily dependent on image quality. Clinics should adopt standard lighting and focal-distance protocols when photographing lesions to match the ISIC training parameters.
3. **Continuous Learning:** As more verified clinical data becomes available, particularly for underrepresented diseases like *Seborrheic keratosis*, the model should be continuously retrained to improve its baseline accuracy.

---

## 🚀 Future Scope & Improvements
While this project effectively establishes a custom CNN from scratch, future iterations could leverage the following:
1. **Transfer Learning:** Implement pre-trained, state-of-the-art architectures like **ResNet50**, **VGG16**, or **EfficientNet**. By utilising weights pre-trained on ImageNet, the model would require significantly less training time and yield higher validation accuracy.
2. **Dynamic Learning Rates:** Integrate callbacks such as `ReduceLROnPlateau` or `EarlyStopping` to automatically adjust the learning rate during training, helping the optimiser find the absolute global minimum without overshooting.
3. **Heatmap Generation (Grad-CAM):** Implement Gradient-weighted Class Activation Mapping to visually highlight the exact pixels/regions of the image that the CNN used to make its classification, providing vital interpretability for doctors.

---

## 📁 Repository Structure

    ├── CNN_Assignment_Vinodh.ipynb          # Primary Jupyter Notebook containing the full pipeline
    ├── vinodh_nagarajaiah_nn.ipynb          # Supplementary neural network notebook
    ├── cnn-assignment-vinodh.pdf            # Detailed report and subjective responses
    ├── cnn_upgrad_vinodh_nagarajaiah.txt    # Submission notes
    └── README.md                            # Project overview and insights

---

## 🎓 Acknowledgements & Contact
This project is an assessment exercise designed and integrated into the AI/ML Programme at **UpGrad**, in collaboration with **IIIT-B**. 

**Created by:** Vinodh Nagarajaiah  

* 💼 **LinkedIn:** [vinodh-nagarajaiah](https://www.linkedin.com/in/vinodh-nagarajaiah/)
* 🐙 **GitHub:** [@techexorcist](https://github.com/techexorcist)
* ✉️ **Email:** [vinodh.nagarajaiah@gmail.com](mailto:vinodh.nagarajaiah@gmail.com)

<br>

> **Disclaimer:** *The dataset used in this project is for educational purposes only. All personally identifiable information (PII) has been removed or anonymised.*

---

## 📜 Licence
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the LICENSE file for details.
