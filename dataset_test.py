from torchvision import transforms , datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split


transform = transforms.Compose([
    transforms.Resize((224 , 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    r"C:\Users\Maryam\Desktop\CatsDogsDataset\PetImages" ,
    transform = transform
)

train_size = 20000
validation_size = 2500
test_size = 2500

train_dataset , validation_dataset , test_dataset = random_split(
    dataset ,
    [train_size , validation_size ,test_size]
) 

print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))

train_loader = DataLoader(
    train_dataset ,
    batch_size = 32 ,
    shuffle = True
)

validation_loader = DataLoader(
    validation_dataset ,
    batch_size = 32 ,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset ,
    batch_size = 32 ,
    shuffle = False
)


images , labels = next(iter(train_loader))
print("Label :" ,labels[0])

image = images[0]

image = image.permute(1 , 2 , 0)

plt.imshow(image)
plt.show()
