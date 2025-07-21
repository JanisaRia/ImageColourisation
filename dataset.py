import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256)):  # Resize to 256x256
        self.root_dir = root_dir
        self.image_filenames = [f for f in os.listdir(
            root_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.image_size = image_size  # Store image size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Resize image to fixed size
        image = cv2.resize(image, self.image_size,
                           interpolation=cv2.INTER_AREA)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L_channel, AB_channels = lab_image[:, :, 0], lab_image[:, :, 1:]

        # Normalize
        L_channel = L_channel / 255.0  # Normalize L
        AB_channels = (AB_channels - 128) / 128  # Normalize AB

        # Convert to tensors
        L_channel = torch.tensor(L_channel, dtype=torch.float32).unsqueeze(
            0)  # Shape: [1, H, W]
        AB_channels = torch.tensor(AB_channels, dtype=torch.float32).permute(
            2, 0, 1)  # Shape: [2, H, W]

        return L_channel, AB_channels


# Set dataset path (UPDATE THIS with your actual dataset path)
dataset_path = r""


# Create dataset instance
dataset = ColorizationDataset(root_dir=dataset_path)

# Load a sample image
L, AB = dataset[0]

# Convert tensors to NumPy for visualization
L_np = (L.squeeze(0).numpy() * 255).astype(np.uint8)  # Denormalize L-channel
AB_np = ((AB.numpy() * 128) + 128).astype(np.uint8)  # Denormalize AB-channels

# Create an empty LAB image
lab_image = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.uint8)

# Assign L and AB channels correctly
lab_image[:, :, 0] = L_np  # Assign L-channel
lab_image[:, :, 1:] = AB_np.transpose(1, 2, 0)  # Fix AB shape before assigning

# Convert LAB to RGB
color_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

# Plot Grayscale and Color Image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(L_np, cmap="gray")
plt.title("Grayscale Input (L-channel)")

plt.subplot(1, 2, 2)
plt.imshow(color_image)
plt.title("Ground Truth Color Image")

plt.show()
