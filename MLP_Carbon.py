import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256), #28*28 = 784 neurons in input layer
            nn.ReLU(),
            nn.Linear(256, 128), #first hidden layer has 256 neurons
            nn.ReLU(),
            nn.Linear(128, 10) #second layer has 128 neurons #output layer has 10 neurons
        )

    def forward(self, x):
        # start_time_input_layer = time.time()
        # x = self.flatten(x)
        # input_time = time.time() - start_time_input_layer
        # # Optionally, print the time taken by the input layer
        # print(f"Time taken by input layer (flattening): {input_time:.6f} seconds")
        # return self.layers(x)
        # Measure time for the first hidden layer and activation
        # Measure time for the input layer (flattening)
        start_time = time.time()
        x = self.flatten(x)
        input_time = time.time() - start_time
        print(f"Time taken by input layer (flattening): {input_time:.6f} seconds")
        
        # Measure time for the first hidden layer and activation
        start_time = time.time()
        x = self.layer1(x)
        x = self.relu1(x)
        hidden_layer1_time = time.time() - start_time
        print(f"Time taken by first hidden layer (Linear + ReLU): {hidden_layer1_time:.6f} seconds")
        
        # Measure time for the second hidden layer and activation
        start_time = time.time()
        x = self.layer2(x)
        x = self.relu2(x)
        hidden_layer2_time = time.time() - start_time
        print(f"Time taken by second hidden layer (Linear + ReLU): {hidden_layer2_time:.6f} seconds")
        
        # Measure time for the output layer
        start_time = time.time()
        x = self.layer3(x)
        output_layer_time = time.time() - start_time
        print(f"Time taken by output layer (Linear): {output_layer_time:.6f} seconds")
        
        return x

# Initialize model and move to GPU if available
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU if available
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Loss: {running_loss/100:.3f} | Acc: {100.*correct/total:.2f}%')
            running_loss = 0
            correct = 0
            total = 0

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    print(f'Test Loss: {test_loss/len(test_loader):.3f} | Test Acc: {100.*correct/total:.2f}%')

# Visualize some images
def show_images(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# Training loop
epochs = 2
for epoch in range(epochs):
    print(f'\nEpoch: {epoch+1}')
    train(model, train_loader, criterion, optimizer, device)
    test(model, test_loader, criterion, device)

# Show some images
# show_images(train_loader)
summary(model, input_size=(1, 28, 28))


# Save model
torch.save(model.state_dict(), 'mnist_mlp.pth')
