# 4. Run Train + สรุปผล + สร้างกราฟ
# -------------------------------


# --- 1.Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')            # Agg backend → ป้องกัน error เมื่อรันบนเครื่องที่ไม่มีหน้าต่าง GUI
import matplotlib.pyplot as plt

from utils import get_dataloaders
from model_ShufflenetV2 import get_model
from training_utils import train_one_epoch, eval_one_epoch
from collections import defaultdict


# --- 2. ฟังก์ชันคำนวณ Per-class Accuracy
def per_class_accuracy(model, dataloader, classes, device):
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name:15s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name:15s}: No samples")


# --- 3. ฟังก์ชัน main()
def main():
    # อุปกรณ์ที่ใช้ประมวลผล (GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # path ของ train and test
    train_dir = "data/Training"
    test_dir = "data/Testing"
    # classes = ["Green_Curry", "Khao_phat", "Khao_Soi", "Massaman_Curry",    -----------  OLD
    #       "Pad_Krapraw", "Pad_Thai", "SomTum", "Tom_yum"]
    classes = [ "Tom_yum", "SomTum", "Pad_Thai", "Pad_Krapraw", "Massaman_Curry",  # ----  NEW ทำการสลับตำแหน่งของคลาส
            "Khao_Soi", "Khao_phat", "Green_Curry"]

    train_loader, test_loader = get_dataloaders(train_dir, test_dir)

# --- 4. สร้างโมเดล Loss function + Optimizer (มาปรับจูนโมเดลตรงนี้นะ)
    model = get_model(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()                      # ใช้ CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # ใช้ Adam optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler.step()                                       # ลดค่า lr ทุก 10 รอบ (step_size=10)

# --- 5. Train + Test แต่ละ epoch
    epochs = 25                    # จำนวนการรันปายยย
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
                      # append ค่า loss & accuracy เพื่อสร้างกราฟทีหลัง

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# --- 6. Per-class Accuracy หลังเทรนเสร็จ
    per_class_accuracy(model, test_loader, classes, device)

# --- 7. Plot results
    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Accuracy graph
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


# --- 8. เรียกใช้งาน main
if __name__ == "__main__":
    main()