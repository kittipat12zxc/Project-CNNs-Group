# 3. เทรนและทดสอบโมเดล พร้อมเก็บค่า loss กับ accuracy
# ต่อไปกด 3. run_shufflenet.py

import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_total = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss_total / len(dataloader), 100 * correct / total
