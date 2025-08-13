import torch
print(torch.__version__)
print(torch.version.cuda)              # ควรขึ้น 12.1
print(torch.cuda.is_available())      # ควรขึ้น True
print(torch.cuda.get_device_name(0))  # ขึ้นชื่อการ์ดจอ
