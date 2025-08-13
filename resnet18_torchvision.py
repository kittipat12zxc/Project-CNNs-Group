import torchvision.models as models
import torch.nn as nn

#โมเดลสำเร็จรูป ResNet18 จาก torchvision
#สามารถเลือกโหลด weight ที่ฝึกมาแล้วหรือไม่ก็ได้
#สามารถเลือก fine-tune หรือไม่ก็ได้
#num_classes คือจำนวนคลาสที่ต้องการจำแนก
#หากต้องการใช้โมเดลนี้ในงานอื่นๆ ควรตั้งค่า
#pretrained=False และ fine_tune=False เพื่อไม่ให้โหลด weight ที่ฝึกมาแล้ว
#และไม่ให้ปรับแต่ง weight ของเลเยอร์ที่ซ่อนอยู่
#หากต้องการใช้โมเดลนี้ในงานจำแนกภาพ ควร
#ตั้งค่า pretrained=True และ fine_tune=True เพื่อใช้ weight ที่ฝึกมาแล้ว      
def build_model(pretrained=True, fine_tune=True, num_classes=8):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif not pretrained:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet18(weights=None)
    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head, it is trainable.
    model.fc = nn.Linear(512, num_classes)
    return model


if __name__ == "__main__":
    model = build_model(num_classes=8)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
