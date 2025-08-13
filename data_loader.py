# 1. โหลดข้อมูลและเตรียม DataLoader
# ต่อไปกด 2. model.py

from torchvision import datasets, transforms
import torch

def get_dataloaders(train_dir, test_dir, batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),   # พลิกภาพ
        transforms.RandomRotation(15),       # หมุนภาพ
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # เปลี่ยนสีเล็กน้อย
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader