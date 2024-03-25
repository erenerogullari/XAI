import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_size=100, hidden_dim=256, output_size=28 * 28):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_dim=256):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    encoder,
    generator,
    g_optimizer,
    e_optimizer,
    train_loader,
    criterion,
    num_epochs=5,
    input_size=100,
):

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            # Flatten MNIST images to a (batch_size, 784) vector
            images = images.view(images.size(0), -1)

            # Create labels for real and fake images
            real_labels = torch.ones(images.size(0), 1)
            fake_labels = torch.zeros(images.size(0), 1)

            # Train Encoder
            e_optimizer.zero_grad()
            outputs = encoder(images)
            e_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Generate fake images
            noise = torch.randn(images.size(0), input_size)
            fake_images = generator(noise)
            outputs = encoder(fake_images.detach())
            e_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            e_loss = e_loss_real + e_loss_fake
            e_loss.backward()
            e_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(images.size(0), input_size)
            fake_images = generator(noise)
            outputs = encoder(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            g_optimizer.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], e_loss: {e_loss.item():.4f}, g_loss: {g_loss.item():.4f}, E(x): {real_score.mean().item():.2f}, E(G(z)): {fake_score.mean().item():.2f}"
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
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
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
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        ),
    )

    return train_dataset, test_dataset


# Function to display images
def show_generated_images(images, num_images=25, title="Generated Images"):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    # Hyperparameters for training
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0002

    # Load data
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize the network
    generator = Generator()
    encoder = Encoder()

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.BCELoss()

    # Train the model
    train_model(
        encoder,
        generator,
        g_optimizer,
        e_optimizer,
        train_loader,
        criterion,
        num_epochs=num_epochs,
    )

    # Generate images
    with torch.no_grad():
        generator.eval()
        fixed_noise = torch.randn(25, 100)  # 25 random noise vectors
        fake_images = generator(fixed_noise).detach().cpu().numpy()

        # Display generated images
        show_generated_images(fake_images)

    # Save the model
    torch.save(generator.state_dict(), "./Activation Maximization/mnist_dgn.pth")
