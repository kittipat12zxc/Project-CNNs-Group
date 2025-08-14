# 3. Train and Test ตัวโมเดล + พร้อมเก็บค่า loss กับ accuracy
# ------------------------------------------------------


import torch
from tqdm import tqdm


# ตอน ----- TRAIN
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0  # correct = จำนวนภาพที่ทายถูก, total = จำนวนภาพทั้งหมด

    for images, labels in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()      # เคลียร์ gradient เก่า
        outputs = model(images)    # forward pass
        loss = criterion(outputs, labels)  # คำนวณ loss
        loss.backward()            # backward pass (คำนวณ gradient)
        optimizer.step()           # อัปเดตน้ำหนักโมเดล

        # ---------- เก็บค่า loss ----------
        running_loss += loss.item()

        # ---------- เก็บค่า accuracy ----------
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # ---------- คำนวณค่าเฉลี่ย ----------
    train_loss = running_loss / len(dataloader)  # loss เฉลี่ยต่อ batch
    train_acc = 100 * correct / total             # accuracy (%) = correct / total * 100

    return train_loss, train_acc  # คืนค่า loss และ accuracy ของ epoch


# ตอน ------ TEST
def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_total = 0.0
    correct, total = 0, 0

    with torch.no_grad():  # ปิด gradient เพื่อประหยัด memory
        for images, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # ---------- เก็บค่า loss ----------
            loss_total += loss.item()

            # ---------- เก็บค่า accuracy ----------
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # ---------- คำนวณค่าเฉลี่ย ----------
    test_loss = loss_total / len(dataloader)
    test_acc = 100 * correct / total

    return test_loss, test_acc  # คืนค่า loss และ accuracy ของ epoch