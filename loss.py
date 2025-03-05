import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model import UNetWithResNet34

# SSIM Loss function


def ssim_loss(pred, target, window_size=11, size_average=True):
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    sigma1_sq = F.avg_pool2d(pred ** 2, window_size,
                             stride=1, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(target ** 2, window_size,
                             stride=1, padding=window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(pred * target, window_size,
                           stride=1, padding=window_size//2) - mu1 * mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

# Perceptual Loss using VGG16


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use first few layers
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[:8]
        self.vgg_layers = nn.Sequential(*list(vgg.children()))
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG parameters

        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # Convert AB channels to LAB format (add dummy L-channel)
        # Create a zero-filled L-channel with shape [batch, 1, H, W]
        L_dummy = torch.zeros_like(pred[:, :1, :, :])

        # Combine L and AB channels → [batch, 3, H, W]
        pred_lab = torch.cat([L_dummy, pred], dim=1)
        target_lab = torch.cat([L_dummy, target], dim=1)  # Same for target

        # Convert LAB to RGB (VGG16 expects RGB)
        pred_rgb = self.lab_to_rgb(pred_lab)
        target_rgb = self.lab_to_rgb(target_lab)

        # Pass RGB images through VGG16 for feature extraction
        pred_features = self.vgg_layers(pred_rgb)
        target_features = self.vgg_layers(target_rgb)

        perceptual_loss = self.mse_loss(pred_features, target_features)
        return perceptual_loss

    def lab_to_rgb(self, lab_tensor):
        """
        Convert a LAB tensor to an RGB tensor.
        """
        lab_numpy = lab_tensor.permute(0, 2, 3, 1).cpu().detach(
        ).numpy()  # Convert to NumPy for OpenCV processing
        lab_numpy[:, :, :, 0] = lab_numpy[:, :, :, 0] * \
            100  # L-channel normalization
        lab_numpy[:, :, :, 1:] = lab_numpy[:, :, :, 1:] * \
            128  # AB-channel normalization

        rgb_list = []
        for i in range(lab_numpy.shape[0]):  # Process batch
            rgb = cv2.cvtColor(lab_numpy[i].astype(
                np.float32), cv2.COLOR_LAB2RGB)
            rgb_list.append(rgb)

        rgb_tensor = torch.tensor(np.stack(rgb_list)).permute(
            0, 3, 1, 2).float().to(lab_tensor.device)
        return rgb_tensor


# Combined Loss Function


class ColorizationLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.2, lambda_ssim=0.2):
        super(ColorizationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = ssim_loss(pred, target)

        total_loss = (self.lambda_mse * mse) + (self.lambda_perceptual *
                                                perceptual) + (self.lambda_ssim * ssim)
        return total_loss


if __name__ == "__main__":
    from loss import ColorizationLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithResNet34().to(device)
    criterion = ColorizationLoss().to(device)  # Loss function

    # Dummy Input & Target
    sample_input = torch.randn(1, 1, 256, 256).to(device)  # Grayscale input
    target_output = torch.randn(1, 2, 256, 256).to(
        device)  # AB color channels (Ground truth)

    # Model Prediction
    pred_output = model(sample_input)

    # Compute Loss
    loss_value = criterion(pred_output, target_output)
    print("Loss Value:", loss_value.item())  # Should be a valid float number
