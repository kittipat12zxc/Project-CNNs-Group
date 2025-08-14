# 1. Transform + โหลด Dataset + สร้าง DataLoader
# ---------------------------------------------


# utils เป็นคำย่อมาจาก “utilities” แปลตรง ๆ คือ “เครื่องมือ/ฟังก์ชันเสริม”
#โค้ดนี้เป็น ฟังก์ชันสำหรับเตรียมข้อมูลภาพ เพื่อให้โมเดล PyTorch สามารถนำไป train และ test

from torchvision import datasets, transforms
import torch

def get_dataloaders(train_dir, test_dir, batch_size=32):

# --- 1. Transform สำหรับ Training
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),                             # 224x224 px (ขนาดมาตรฐานของโมเดล ShuffleNet)
        transforms.RandomHorizontalFlip(),                         # พลิกภาพ
        transforms.RandomRotation(15),                             # หมุนภาพ 15 องศา
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # เปลี่ยนสีเล็กน้อย
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # # Normalize แต่ละช่อง R,G,B  (ค่า pixel ให้มี mean=0.5, std=0.5)
]) 
    
# --- 2. Transform สำหรับ Testing 
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

# --- 3. โหลด Dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

# --- 4. สร้าง DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32,                                # กำหนดจำนวนภาพต่อ batch
        shuffle=True,                                 # สุ่มภาพทุก epoch เพื่อให้โมเดลเรียนรู้แบบไม่จำเจ
        
        num_workers=4)                                # จำนวน subprocess ที่ใช้โหลดภาพ (ช่วยให้โหลดเร็วขึ้น)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,                                # ไม่ต้อง shuffle ข้อมูล test
        num_workers=4)

# --- 5. คืนค่า DataLoader
    return train_loader, test_loader