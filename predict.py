import torch
from torchvision import transforms
from PIL import Image

from model import CatDogCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CatDogCNN().to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(
    r"C:\Users\Maryam\Downloads\Comparison-of-German-dog-with-Doberman-2.jpg"
)

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

output = model(image)

probabilities = torch.softmax(output, dim=1)

cat_prob = probabilities[0][0].item()
dog_prob = probabilities[0][1].item()
    
print(f"Cat: {cat_prob * 100:.2f}%")
print(f"Dog: {dog_prob * 100:.2f}%")
