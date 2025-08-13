# 2. สร้างและคืนค่าโมเดล
# ต่อไปกด 3. train.py

import torch.nn as nn
from torchvision import models

def get_model(num_classes=8):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
