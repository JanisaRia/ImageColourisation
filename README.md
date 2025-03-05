# **Deep Learning-Based Image Colorization using U-Net with ResNet-34 Encoder**

## **Overview**
This project implements an **image colorization model** that converts grayscale images into realistic color images using **deep learning**. The model is based on a **U-Net architecture with a ResNet-34 encoder** and utilizes **LAB color space conversion** for effective learning. **Perceptual loss (VGG16), MSE loss, and SSIM loss** are used to enhance color accuracy.

---

## **Dataset**
- The dataset consists of colored images, which are converted into grayscale (L-channel) during training.
- The dataset used: [Image Colorization Dataset - Kaggle](https://www.kaggle.com/datasets/mertbozkurt5/image-colorization/data).

---

## **Architecture**
- **U-Net** with **ResNet-34** encoder (pretrained on ImageNet)
- **Self-Attention Mechanisms:** Squeeze-and-Excitation (SE-Blocks) or CBAM
- **Loss Functions:** 
  - **MSE Loss** (Pixel-wise similarity)
  - **Perceptual Loss** (Feature-level similarity using VGG16)
  - **SSIM Loss** (Structural similarity)
- **LAB Color Space Conversion:** 
  - L-channel as input (grayscale)
  - Predicted AB channels as output (color information)

---

## **Tools & Technologies**
- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** OpenCV, Matplotlib
- **Optimization:** Adam Optimizer with Learning Rate Scheduling
- **Dataset Handling:** PyTorch Dataset & DataLoader
- **Image Processing:** LAB Color Space Conversion, Bilateral Filtering

---

## **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/image-colorization.git
cd image-colorization
