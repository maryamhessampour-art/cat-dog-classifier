import torch.nn as nn


class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()  
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(93312 , 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128 , 2)
        
    def forward(self , x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
        

import torch

model = CatDogCNN()

x = torch.randn(1, 3, 224, 224)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)