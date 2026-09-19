import torch 
from torch.utils.data import DataLoader , random_split
from torchvision import transforms , datasets
import torch.nn as nn
from model import CatDogCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    r"C:\Users\Maryam\Desktop\Cat_Dog_Classifier\dataset\train",
    transform=transform
)

print("Train dataset size:", len(train_dataset))
print("Classes:", train_dataset.classes)

total_train_size = len(train_dataset)



train_size = int(0.9 * total_train_size)
validation_size = total_train_size - train_size



train_dataset, validation_dataset = random_split(
    train_dataset,
    [train_size, validation_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)



validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)


model = CatDogCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters() ,
    lr = 0.001
)

num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    
    model.train()
    
    running_loss = 0.0

    for images , labels in train_loader :
    
       images = images.to(device)
       labels = labels.to(device)

       optimizer.zero_grad()
 
       outputs = model(images)

       loss = criterion(outputs, labels)

       loss.backward()

       optimizer.step()
    
       running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Loss: {epoch_loss:.4f}")
    
    
    
    
    
    model.eval()
    
    correct = 0 
    total = 0
    validation_loss = 0.0
    
    with torch.no_grad():
        
        for images , labels in validation_loader :
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs , labels)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
            
            validation_loss += loss.item()
            
    accuracy = 100 * correct / total
    
    avg_validation_loss = validation_loss / len(validation_loader)
    
    if accuracy > best_accuracy :
        best_accuracy = accuracy
        torch.save(model.state_dict() , "best_model.pth")
        print("Best model saved!")
    
    print(
    f"Validation Loss: {avg_validation_loss:.4f}, "
    f"Validation Accuracy: {accuracy:.2f}%"
    )
    
cat = 0
dog = 0

for _, label in train_dataset:
    if label == 0:
        cat += 1
    else:
        dog += 1

print("Train Cat:", cat)
print("Train Dog:", dog)