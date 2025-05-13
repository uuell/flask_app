import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# TODO's
    # from fakeImage import TinyCNN

    # A train/test split?

    # Expanding the model to make it a little deeper?

    # Saving the model?

    # Try loading a famous or already built model?

# Transform: convert image to tensor and resize it
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(root='images', transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Show class names (just to confirm)
print("Classes:", dataset.classes)  # Should show ['cats', 'dogs']

# 2. Build a small CNN
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: 2 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = TinyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
print("Training complete!")

# Get a batch of data
dataiter = iter(loader)
images, labels = next(dataiter)

# Predict
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Show predictions
for i in range(2):  # You only have 2 images per class
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Predicted: {dataset.classes[predicted[i]]} | Actual: {dataset.classes[labels[i]]}")
    plt.axis('off')
    plt.show()