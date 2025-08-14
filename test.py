import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from collections import defaultdict

def main():
    test_dir = "data/Testing"
    model_path = "resnet_torchvision.pth" # Path to the trained model weights
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [
        "Green_Curry",
        "Khao_Phat",
        "Khao_Soi",
        "Massaman_Curry",
        "Pad_Krapraw",
        "Pad_Thai",
        "Som_Tum",
        "Tom_Yum"
       
    ]
    num_classes = len(class_names)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    print("Classes in dataset:", idx_to_class)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    total = 0
    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1

    overall_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\nOverall accuracy: {overall_acc:.2f}%  ({correct}/{total})")

    print("\nPer-class accuracy:")
    for idx in sorted(idx_to_class.keys()):
        cname = idx_to_class[idx]
        tot = per_class_total[idx]
        corr = per_class_correct[idx]
        acc = 100.0 * corr / tot if tot > 0 else 0.0
        print(f" - {cname}: {acc:.2f}%  ({corr}/{tot})")

if __name__ == '__main__':
    main()
