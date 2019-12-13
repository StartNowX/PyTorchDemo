import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import cv2

def LoadingCIFAR10():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root="./data/", 
                                            train=True, 
                                            download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root="./data",
                                           train=False,
                                           download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print("\t In model: input size", a.size(),
                # "output size", x.size())
        return x


def SaveImage(img):
    img = img / 2 + 0.5
    npimg = img.numpy() * 255
    cv2.imwrite("./data/img.jpg", np.transpose(npimg, (1, 2, 0)))

def TrainClassifier():
    multi_gpus = True
    gpu_ids = [3, 4]
    if multi_gpus:
        # cuda: 0, 代表显卡的起始编号是0, gpu_ids的第一个卡的编号要与之相同
        device = torch.device("cuda:{}".format(gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    ## 另一种方式使用指定的多卡
    ## import os
    ## os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

    trainloader, testloader, classes = LoadingCIFAR10()

    net = Net();
    if multi_gpus:
        ## 如果如此进行训练，测试时模型也需要用nn.DataParallel
        net = nn.DataParallel(net, device_ids=gpu_ids).to(device)
    else:
        net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Begin Training.")
    for epoch in range(2):

        running_loss = 0.0
        for idx, data in enumerate(trainloader, 0):
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 2000 == 1999:
                print("Outside: input size ", inputs.size(), 
                    "output size", outputs.size())
                print("[%d, %5d] loss: %.3f" % (epoch + 1, idx + 1, running_loss / 2000))
                running_loss = 0.0
            
    print("Finish Training")

    savepath = "./model/cifar_net.pth"
    torch.save(net.state_dict(), savepath)


def TestClassifier(modelpath):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    multi_gpus = True
    gpu_ids = [3]

    net = Net()
    if multi_gpus:
        net = nn.DataParallel(net, device_ids=gpu_ids).to(device)
    else:
        net.to(device)
    trainloader, testloader, classes = LoadingCIFAR10()

    net.load_state_dict(torch.load(modelpath))

    correct = 0
    total = 0

    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for idx in range(4):
                label = labels[idx]
                class_correct[label] += c[idx].item()
                class_total[label] += 1

    for i in range(10):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

if __name__ == "__main__":
    trainloader, testloader, classes = LoadingCIFAR10()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print("images: ", images.size())

    SaveImage(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # TrainClassifier()

    modelpath = "./model/cifar_net.pth"
    TestClassifier(modelpath)