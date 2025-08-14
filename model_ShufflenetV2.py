# 2. สร้างโมเดล ShuffleNetV2 และคืนค่าโมเดล
# --------------------------------------


import torch.nn as nn
from torchvision import models

def get_model(num_classes=8):
    model = models.shufflenet_v2_x1_0(pretrained=True)      # -- โหลดโมเดล ShuffleNetV2 ขนาด x1.0 พร้อม weights ที่ pretrained บน ImageNet
    num_ftrs = model.fc.in_features                         # -- ดึงจำนวน input features ของ fully connected layer เดิม
    model.fc = nn.Linear(num_ftrs, num_classes)             # -- สร้าง fully connected layer ใหม่ ให้ output เท่ากับจำนวน class ของเรา

    # -- คืนค่าโมเดลที่ปรับแล้ว
    return model

# Flow : Pretrained ShuffleNetV2 → Replace fc → Output layer = num_classes → Return model