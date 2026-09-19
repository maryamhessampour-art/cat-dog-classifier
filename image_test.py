from PIL import Image
import torchvision.transforms as transforms

image = Image.open(r"C:\Users\Maryam\Desktop\CatsDogsDataset\PetImages\Cat\1.jpg")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_tensor = transform(image)

print("Original size:", image.size)
print("Tensor shape:", image_tensor.shape)