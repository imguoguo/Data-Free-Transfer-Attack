import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# 定义保存预训练模型的目录
pretrained_dir = 'pretrained_ckpt'
os.makedirs(pretrained_dir, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 通用训练函数
def train_model(model, train_loader, num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    return model

class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# 各数据集的训练和保存代码
def save_mnist_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = CNN()
    model = train_model(model, train_loader, num_epochs=5)
    torch.save({'state_dict': model.state_dict()}, os.path.join(pretrained_dir, 'cnn_mnist.pth'))

def save_fmnist_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = CNN()
    model = train_model(model, train_loader, num_epochs=5)
    torch.save({'state_dict': model.state_dict()}, os.path.join(pretrained_dir, 'cnn_fmnist.pth'))

def save_cifar10_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = CNNCifar10()
    model = train_model(model, train_loader, num_epochs=10)
    torch.save({'state_dict': model.state_dict()}, os.path.join(pretrained_dir, 'cnncifar10.pkl'))

def save_svhn_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = models.resnet18(num_classes=10)
    model = train_model(model, train_loader, num_epochs=10)
    torch.save({'state_dict': model.state_dict()}, os.path.join(pretrained_dir, 'res18_svhn.pth'))

def save_cifar100_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = models.resnet50(num_classes=100)
    model = train_model(model, train_loader, num_epochs=10)
    torch.save({'state_dict': model.state_dict()}, os.path.join(pretrained_dir, 'res50_cifar100.pth'))

def save_tiny_imagenet_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = models.resnet50(num_classes=200)
    model = train_model(model, train_loader, num_epochs=10)
    torch.save(model.state_dict(), os.path.join(pretrained_dir, 'res50_tiny_imagenet.pth'))

# 调用每个函数以训练并保存模型
#save_mnist_model()
#save_fmnist_model()
save_cifar10_model()
save_svhn_model()
save_cifar100_model()
# save_tiny_imagenet_model()

print("所有模型都已训练并保存在 'pretrained_ckpt' 目录中。")
