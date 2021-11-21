import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
import requests
from PIL import Image

DIR = '/applications/pytorch-study'
trainDIR = DIR + '/train'
testDIR = DIR + '/test'
epoches = 10
batch_size = 32
args1 = [16,32,64,128]
image_size = 30

for i in range(len(args1)):
    image_size = image_size*2+4
print(image_size)

trans = transforms.Compose([
                           transforms.Resize((image_size,image_size)),
                           transforms.ToTensor(),
                           ])
train_set = torchvision.datasets.ImageFolder(root=trainDIR,transform=trans)
train_loader = DataLoader(train_set,shuffle=True, batch_size=batch_size)

classes = sorted(os.listdir('/applications/pytorch-study/train')[1:])
l = len(classes)

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,args1[0],5)
        self.conv2 = nn.Conv2d(args1[0],args1[1],5)
        self.conv3 = nn.Conv2d(args1[1],args1[2],5)
        self.conv4 = nn.Conv2d(args1[2],args1[3],5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 30 * 30, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, l)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = myCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
l2 = len(train_loader)
ifTrained = True
if ifTrained:
    net = torch.load('model-cnnWith4convs.pt')


def train():
    for epoch in range(epoches):
        running_losses = 0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            print('\r' + '当前epoch的进度：%.1f' % (100 * (i + 1) / l2), '%', end='')
            inputs,train_label = data
            outputs = net(inputs)
            optimizer.zero_grad()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == train_label.data).sum()
            loss = criterion(outputs, train_label)
            loss.backward()
            optimizer.step()
            running_losses += loss.item()
            train_total += train_label.size(0)
            print()
            print('train %d epoch loss: %.3f  acc: %.3f ' % (
                epoch + 1, running_losses / train_total, 100 * train_correct / train_total))
        torch.save(net, 'model-cnnWith4convs.pt')


def showImage():
    inputs = trans(Image.open('/applications/pornpics/nicolette shea/12013066_001_16ea.jpg'))
    inputs = torch.unsqueeze(inputs, dim=0)
    net2 = torch.load('model-cnnWith4convs.pt')
    out = net2(inputs)
    _,predicted = torch.max(out.data,1)
    print('预测是', classes[predicted.numpy()[0]],'实际是shea')
    inputs = torchvision.utils.make_grid(inputs)
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.show()


def test():
    net = torch.load('model-cnnWith4convs.pt')
    net.eval()





