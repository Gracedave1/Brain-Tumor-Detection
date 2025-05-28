# Brain-Tumor-Detection
# 🧠 Brain Tumor Detection Using Deep Learning (CNN-Based Approach)

## 📌 Project Overview

The early detection of brain tumors can be life-saving, but manual analysis of MRI scans is time-consuming and prone to error. This project aims to leverage **Convolutional Neural Networks (CNNs)** to build a robust, accurate, and automated diagnostic model capable of distinguishing between healthy and tumorous brain scans.

This deep learning pipeline streamlines the detection of tumors in MRI images by transforming raw pixel data into diagnostic insights - all through an end-to-end model embedded in an interactive web application.

---

## 📂 Dataset Description

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- **Total Images**: **7,023 MRI scans**
- **Format**: PNG, grayscale and RGB images
- **Categories** (originally multi-class):
  - `Glioma Tumor`
  - `Meningioma Tumor`
  - `Pituitary Tumor`
  - `No Tumor`

🧪 **Classification Goal**: Transformed into a **binary classification problem**:
- **Tumor**: Any of the three tumor classes (Glioma, Meningioma, Pituitary)
- **No Tumor**: Healthy brain scan

---

## 🛠️ Methodology

### 1. 🧹 Data Preprocessing
To prepare the dataset for deep learning, the following preprocessing steps were applied:
- **Resizing** all images to a uniform dimension of `150x150` pixels
- **Normalization** of pixel intensities to a `[0, 1]` scale for improved convergence
- **RGB Conversion**: Ensured consistent color channels across all images
- **Shuffling & Splitting**: Stratified split into training, validation, and testing sets

### 2. 🧪 Data Augmentation
To increase the dataset's variability and prevent overfitting, extensive data augmentation was employed:
- Horizontal and vertical **flipping**
- **Random rotation** (up to 20 degrees)
- **Zooming** and **shifting**
- Minor **brightness and contrast adjustments**

This synthetic diversity allowed the CNN to generalize better across different tumor types and orientations.

### 3. 🧠 CNN Model Architecture

A custom-built **Convolutional Neural Network** (CNN) was designed for efficient feature extraction and classification:

- 3 × **Convolutional layers** with ReLU activation and max pooling
- 1 × **Flattening layer**
- 2 × **Fully connected (Dense) layers**
- **Dropout layers** to reduce overfitting
- Final output: **Sigmoid** layer for binary classification

> ⚙️ **Optimizer**: Adam  
> 🔁 **Loss Function**: Binary Crossentropy  
> 📉 **Training Strategy**: Mini-batch Gradient Descent (batch size = 32)

---

## 📊 Performance Metrics

| Metric             | Value         |
|--------------------|---------------|
| **Training Accuracy** | 96.72%      |
| **Validation Accuracy** | 95.19%   |
| **Test Accuracy**     | **95.50%**  |
| **Loss (Test)**       | Low & stable |
| **Inference Time**    | Real-time (< 1s)

These results indicate that the model is both **accurate** and **generalizes well** to unseen MRI scans.

---

## 🚀 Deployment

To make this model accessible for practical use:

- Integrated into a **real-time Streamlit web application**
- Users can **upload MRI images**, and the model will return an instant **tumor vs. no tumor prediction**
- Clean, user-friendly UI with confidence score display

---

## 🧾 Conclusion & Insights

- The CNN model demonstrates strong potential as an assistive tool for **radiologists and clinicians**, achieving over **95% accuracy** on test data.
- With minimal latency and intuitive deployment, the system can significantly **accelerate the diagnostic process** in real-world clinical workflows.
- Future enhancements could include:
  - Multi-class classification to identify **tumor type**
  - Integration with **patient metadata** (age, gender, symptoms)
  - **Explainable AI** overlays (e.g., Grad-CAM) to highlight tumor regions

---

## 💡 Key Takeaways

- ✅ Deep learning can be highly effective in medical imaging diagnostics with proper preprocessing and augmentation.
- ✅ Binary classification (tumor vs. no tumor) is a viable entry point for real-time tumor screening.
- ✅ Deployment via Streamlit bridges the gap between machine learning and real-world usability.

---
