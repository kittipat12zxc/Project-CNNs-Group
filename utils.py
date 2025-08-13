import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets , transforms
from torchvision.transforms import ToTensor,Compose, Resize
from torchvision.datasets import ImageFolder

plt.style.use("ggplot")


def get_data(batch_size=64):
    # Transforms (คุณจะเพิ่ม augment ได้ในภายหลัง)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # เพราะ ResNet18 ต้องใช้ input 224x224
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    ])

    # Load custom training and validation datasets
    dataset_train = datasets.ImageFolder(
        root="data/Training", #  เปลี่ยนเป็น path ของ training set ของคุณ
        transform=transform
    )

    dataset_valid = datasets.ImageFolder(
        root="data/Testing",# เปลี่ยนเป็น path ของ validation set ของคุณ
        transform=transform
    )

    # Create data loaders.
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="tab:red", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_accuracy.png"))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="tab:red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_loss.png"))
