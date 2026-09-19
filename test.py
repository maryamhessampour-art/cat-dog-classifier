import torch

from torch.utils.data import DataLoader

from torchvision import transforms, datasets

import torch.nn as nn

from model import CatDogCNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


test_dataset = datasets.ImageFolder(
    r"C:\Users\Maryam\Desktop\Cat_Dog_Classifier\dataset\test",
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


model = CatDogCNN().to(device)

criterion = nn.CrossEntropyLoss()


model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)


model.eval()


test_loss = 0.0

correct = 0

total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        test_loss += loss.item()


avg_test_loss = test_loss / len(test_loader)

test_accuracy = 100 * correct / total


print(f"Test Loss: {avg_test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.2f}%")

cat = 0
dog = 0

for _, label in test_dataset:
    if label == 0:
        cat += 1
    else:
        dog += 1

print("Test Cat:", cat)
print("Test Dog:", dog)