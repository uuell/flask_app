import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Create a fake image dataset (like real images, 2 classes)
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.FakeData(
    size=200, image_size=(3, 64, 64), num_classes=2, transform=transform
)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

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

# 3. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Get one batch and predict
dataiter = iter(train_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Show first 4 images
for i in range(4):
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted[i].item()} | Actual: {labels[i].item()}")
    plt.show()