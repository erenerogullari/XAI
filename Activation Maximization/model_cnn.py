import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        hidden_sizes=[128, 64],
        kernel_size=3,
        stride=1,
        padding=1,
        fc_input_size=64 * 7 * 7,
    ):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        fc_layers = []
        last_size = fc_input_size
        for size in hidden_sizes:
            fc_layers.extend([nn.Linear(last_size, size), nn.ReLU(inplace=True)])
            last_size = size
        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        # Apply conv + relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        # Apply output layer
        x = self.fc_out(x)

        return x


# Train function
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


# Evaluate function
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )


def load_data():
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Load data
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Define model
    model = ConvNet()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train model
    train_model(model, train_loader, criterion, optimizer, num_epochs=3)

    # Evaluate model
    evaluate_model(model, test_loader)

    # Save the model
    torch.save(model.state_dict(), "mnist_cnn.pth")
