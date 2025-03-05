import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNetWithResNet34
from loss import ColorizationLoss
from dataset import ColorizationDataset

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
# Change this to the actual dataset path
dataset_path = r"C:\Users\janis\learning\machine learning\image colorisatiion\dataset"
dataset = ColorizationDataset(root_dir=dataset_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize Model, Loss, and Optimizer
model = UNetWithResNet34().to(device)
criterion = ColorizationLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)  # Decay LR every 10 epochs

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for L, AB in dataloader:
        L, AB = L.to(device), AB.to(device)

        optimizer.zero_grad()
        output = model(L)

        loss = criterion(output, AB)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the model after every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Model saved: model_epoch_{epoch+1}.pth")

print("Training Complete!")
