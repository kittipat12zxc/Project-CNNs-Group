# แบบรวมไม่แยกไฟล์

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# อุปกรณ์ที่ใช้ประมวลผล (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# path ของ train and test
train_dir = "C:/Github/ResNet18gameii/TrainModelnaja/data/Training"
test_dir = "C:/Github/ResNet18gameii/TrainModelnaja/data/TestIng"

# Data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),             # ShuffleNetV2 ใช้ขนาดภาพเท่านี้
    transforms.RandomHorizontalFlip(),         # สุ่มพลิกภาพ (augmentation) เพื่อให้เรียนรู้ได้ดีขึ้น
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize ค่า pixel
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# โหลดข้อมูลด้วย ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# โหลดโมเดล ShuffleNetV2 + ปรับให้ตรงคลาสด้วยนะจ๊ะ
model = models.shufflenet_v2_x1_0(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # เปลี่ยน output layer ให้ตรงกับคลาส
model = model.to(device)

# Loss function + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ตัวแปรเก็บค่าผลลัพธ์สำหรับกราฟ
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

# Train & Test
epochs = 5
for epoch in range(epochs):
    # ---------- Train ----------
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    print(f"\nEpoch [{epoch+1}/{epochs}] - Training...")
    for images, labels in tqdm(train_loader, desc="Training Progress", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # เคลียร์ gradient เก่า
        outputs = model(images)            # ส่งข้อมูลเข้าโมเดล
        loss = criterion(outputs, labels)  # คำนวณ loss
        loss.backward()                    # คำนวณ gradient
        optimizer.step()                   # อัปเดตน้ำหนักโมเดล

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # ---------- Test ----------
    model.eval()
    test_loss_total = 0.0
    correct_test, total_test = 0, 0

    print(f"Epoch [{epoch+1}/{epochs}] - Testing...")
    for images, labels in tqdm(test_loader, desc="Testing Progress", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():  # ปิด gradient ช่วยให้เร็วขึ้น
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = test_loss_total / len(test_loader)
    test_acc = 100 * correct_test / total_test

    # เก็บค่าไว้สำหรับทำกราฟ
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # ---------- สรุปผลในแต่ละ epoch ----------
    print(f"Epoch [{epoch+1}/{epochs}] Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")


# สร้างกราฟหลังเทรนเสร็จ
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

# Save png the loss and accuracy plots.
plt.tight_layout()
plt.savefig("training_graph.png")
print("Saved training_graph.png")