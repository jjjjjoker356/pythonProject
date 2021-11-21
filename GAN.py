import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

image_size = 28
input_dim = 100
num_channels = 1
num_features = 64
batch_size = 64
use_cuda = torch.cuda.is_available()
dtype = torch.FloatTensor
itype = torch.LongTensor

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            )
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                        )
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]

sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               sampler=sampler_val,
                               )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         sampler=sampler_test,
                         )

class ModelG(nn.Module):
    def __init__(self):
        super(ModelG, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('deconv1', nn.ConvTranspose2d(input_dim,num_features*2, 5, 2, 0, bias=False))
        self.model.add_module('bnorm1', nn.BatchNorm2d(num_features*2))
        self.model.add_module('relu1', nn.ReLU(True))

        self.model.add_module('deconv2', nn.ConvTranspose2d(num_features*2,num_features,5,2,0,bias=False))
        self.model.add_module('bnorm2', nn.BatchNorm2d(num_features))
        self.model.add_module('relu2', nn.ReLU(True))
        self.model.add_module('deconv3', nn.ConvTranspose2d(num_features,num_channels,4,2,0,bias=False))
        self.model.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        output = input
        for name, module in self.model.named_children():
            output = module(output)
        return(output)

def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('conv')!=-1:
        m.weight.data.normal_(0, 0.02)
    if class_name.find('norm')!=-1:
        m.weight.data.normal_(1.0, 0.02)


def make_show(img):
    # 将张量变成可以显示的图像
    img = img.data.expand(batch_size, 3, image_size, image_size)
    return img


def imshow(inp, title=None):
    # 在屏幕上绘制图像
    """Imshow for Tensor."""
    if inp.size()[0] > 1:
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = inp[0].numpy()
    mvalue = np.amin(inp)
    maxvalue = np.amax(inp)
    if maxvalue > mvalue:
        inp = (inp - mvalue)/(maxvalue - mvalue)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


print('Initialized!')
net = ModelG()
net = net.cuda() if torch.cuda.is_available() else net
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
samples = np.random.choice(10, batch_size)
sample = torch.from_numpy(samples)
step=0
num_epoches=100
record=[]

for epoch in range(num_epoches):
    train_loss = []
    for batch_idx, inputs in enumerate(train_loader):
        #target, data = data.clone().detach().requires_grad_(True), target.clone().detach()  # data为一批图像，target为一批标签
        target, data = inputs
        if torch.cuda.is_available():
            target,data = target.cuda(),data.cuda()
        #print(data)
        data = data.type(dtype)

        data = data.resize(data.size()[0], 1, 1, 1)
        data = data.resize(data.size()[0],1,1,1)
        data = data.expand(data.size()[0],input_dim,1,1)
        net.train()
        output = net(data)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if torch.cuda.is_available():
            loss = loss.cpu()
        train_loss.append(loss.data.numpy())

        if step % 100==0:
            net.eval()
            val_loss = []
            idx=0
            for inputs in validation_loader:
                target,data = inputs
                idx+=1
                if torch.cuda.is_available():
                    target, data = target.cuda(), data.cuda()
                data = data.resize(data.size()[0], 1, 1, 1)
                data = data.expand(data.size()[0], input_dim, 1, 1)
                output = net(data)
                loss = criterion(output,target)
                if torch.cuda.is_available():
                    loss = loss.cpu()
                val_loss.append(loss.data.numpy())
            print('训练周期: {} [{}/{} ({:.0f}%)]\t训练数据Loss: {:.6f}\t校验数据Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), np.mean(train_loss), np.mean(val_loss)))
            record.append([np.mean(train_loss), np.mean(val_loss)])
    with torch.no_grad():
        samples.resize_(batch_size, 1, 1, 1)
    samples = samples.data.expand(batch_size, input_dim, 1, 1)
    samples = samples.cuda() if use_cuda else samples  # 加载到GPU
    fake_u = net(samples)  # 用原始网络作为输入，得到伪造的图像数据
    fake_u = fake_u.cpu() if use_cuda else fake_u
    img = make_show(fake_u)  # 将张量转化成可绘制的图像
    os.makedirs('temp1', exist_ok=True)
    torchvision.utils.save_image(img, 'temp1/fake%s.png' % (epoch))  # 保存生成的图像
