# 4. รันเทรน สรุปผล และสร้างกราฟ

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from model import get_model
from train import train_one_epoch, eval_one_epoch

def main():
    # อุปกรณ์ที่ใช้ประมวลผล (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # path ของ train and test
    train_dir = "C:/Github/ResNet18gameii/TrainModelnaja/data/Training"
    test_dir = "C:/Github/ResNet18gameii/TrainModelnaja/data/TestIng"

    train_loader, test_loader = get_dataloaders(train_dir, test_dir)

    # Loss function + Optimizer
    model = get_model(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler.step() # โมเดลจะ fine-tune น้ำหนักได้ดีขึ้น

    # Test
    epochs = 25 # จำนวนการรันปายยย
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_graph.png")
    print("Saved training_graph.png")

if __name__ == "__main__":
    main()