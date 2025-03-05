import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UNetWithResNet34(nn.Module):
    def __init__(self):
        super(UNetWithResNet34, self).__init__()

        # Load Pretrained ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept 1-channel input instead of 3-channel RGB
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder[0] = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Attention Mechanism (CBAM)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoder (Upsampling layers)
        self.upsample1 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1)
        self.upsample5 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1)  # NEW Upsampling Layer
        self.final_layer = nn.Conv2d(
            16, 2, kernel_size=3, padding=1)  # Output AB channels

    def forward(self, x):
        x = self.encoder(x)

        # Apply Attention Mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Decoder (upsampling)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)  # NEW Upsampling Layer
        x = self.final_layer(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithResNet34().to(device)
    print("Model initialized and moved to device:", device)

    # Test with a dummy grayscale image
    # Batch size=1, 1 grayscale channel, 256x256 image
    sample_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(sample_input)

    print("Input shape:", sample_input.shape)  # Expected: [1, 1, 256, 256]
    # Expected: [1, 2, 256, 256] (AB channels)
    print("Output shape:", output.shape)
