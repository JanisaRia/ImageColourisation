import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import UNetWithResNet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetWithResNet34().to(device)
model.load_state_dict(torch.load("model_epoch_10.pth", map_location=device))
model.eval()

image_path = "dataset/2.bengali-village.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Error: The file '{image_path}' was not found. Check the path.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
L_channel = lab_image[:, :, 0]
AB_channels = lab_image[:, :, 1:]

L_tensor = torch.tensor(L_channel / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred_AB = model(L_tensor).cpu().squeeze(0).numpy()

pred_AB = np.clip(pred_AB * 128, -128, 127).astype(np.int8)

print("Predicted AB Min:", pred_AB.min(), "Max:", pred_AB.max())

pred_AB_A = cv2.bilateralFilter(pred_AB[0].astype(np.float32), 9, 75, 75)
pred_AB_B = cv2.bilateralFilter(pred_AB[1].astype(np.float32), 9, 75, 75)
pred_AB_filtered = np.stack([pred_AB_A, pred_AB_B], axis=0).astype(np.int8)

predicted_lab = np.zeros((256, 256, 3), dtype=np.int8)
predicted_lab[:, :, 0] = L_channel
predicted_lab[:, :, 1:] = pred_AB_filtered.transpose(1, 2, 0)

predicted_rgb = cv2.cvtColor(predicted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
predicted_rgb = np.clip(predicted_rgb, 0, 255).astype(np.uint8)

ground_truth_lab = np.zeros((256, 256, 3), dtype=np.uint8)
ground_truth_lab[:, :, 0] = L_channel
ground_truth_lab[:, :, 1:] = AB_channels
ground_truth_rgb = cv2.cvtColor(ground_truth_lab, cv2.COLOR_LAB2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(L_channel, cmap="gray")
plt.title("Grayscale Input (L-channel)")

plt.subplot(1, 3, 2)
plt.imshow(ground_truth_rgb)
plt.title("Ground Truth Color Image")

plt.subplot(1, 3, 3)
plt.imshow(predicted_rgb)
plt.title("Predicted Colorized Image")

plt.show()
