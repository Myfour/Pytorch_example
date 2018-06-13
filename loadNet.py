import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决图片截断的问题


# 1号模型
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.c1 = nn.Sequential(  # (3,100,100)
            nn.Conv2d(3, 6, 5, 1, 2),  # (6,100,100)
            nn.ReLU())
        self.s2 = nn.MaxPool2d(2)  # (6,50,50)
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  # (16,46,46)
            nn.ReLU())
        self.s4 = nn.MaxPool2d(2)  # (16,23,23)
        self.c5 = nn.Linear(16 * 23 * 23, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = x.view(x.size(0), -1)
        x = self.c5(x)
        x = self.f6(x)
        output = self.out(x)
        return output


# 2号模型
class LeNet_5_p(nn.Module):
    def __init__(self):
        super(LeNet_5_p, self).__init__()
        self.c1 = nn.Sequential(  # (3,100,100)
            nn.Conv2d(3, 6, 5, 1, 2),  # (6,100,100)
            nn.ReLU())
        self.s2 = nn.MaxPool2d(2)  # (6,50,50)
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  # (16,46,46)
            nn.ReLU())
        self.s4 = nn.MaxPool2d(2)  # (16,23,23)
        self.c5 = nn.Sequential(
            nn.Conv2d(16, 25, 5, 1),  # (25,19,19)
            nn.ReLU())
        self.s6 = nn.MaxPool2d(2)  # (25,9,9)
        self.f7 = nn.Linear(25 * 9 * 9, 120)
        self.f8 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.s6(x)
        x = x.view(x.size(0), -1)
        x = self.f7(x)
        x = self.f8(x)
        output = self.out(x)
        return output


# 测试集
test_data = torchvision.datasets.ImageFolder(
    '/home/nifer/Desktop/TheFinalDoc/测试集',
    transform=transforms.Compose([
        transforms.Resize((100, 100)),  # 注意使用方式,若只定义一个int值,则是将一个较小维度变为这个值
        # transforms.CenterCrop(
        #    500),  # 将图片从中心裁剪,参数若为一个int类型则裁剪成长度为int类型值的正方形,若为(h,w)则为相应矩形
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))


def load_Mynet(netName):
    if os.path.exists('/home/nifer/VSCode/Pytorch_Go/' + netName + '.pkl'):
        net = torch.load('/home/nifer/VSCode/Pytorch_Go/' + netName + '.pkl')
        print(net)
        return net
    else:
        if netName == 'lenet5':
            net = LeNet_5()
        if netName == 'lenet5_p':
            net = LeNet_5_p()
        return net


# 训练集
img_data = torchvision.datasets.ImageFolder(
    '/home/nifer/Desktop/TheFinalDoc/图片集',
    transform=transforms.Compose([
        transforms.Resize((100, 100)),  # 注意使用方式,若只定义一个int值,则是将一个较小维度变为这个值
        # transforms.CenterCrop(
        #    500),  # 将图片从中心裁剪,参数若为一个int类型则裁剪成长度为int类型值的正方形,若为(h,w)则为相应矩形
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))


def imshow(img):
    img = img / 2 + 0.5  # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def testModel(net):
    '''
    测试训练的结果
    '''
    data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True)
    correct = 0
    total = 0
    for data in data_loader:
        images, labels = data  # 每次遍历得到的是数据集批训练个数的集合
        outputs = net(Variable(images))  # 利用现有参数得到的输出，这里的output得到的是100×10矩阵
        _, predicted = torch.max(outputs.data,
                                 1)  # torch.max()获取输入张量的最大值，加入dim参数后，
        # 返回的是在给定维度上的最大值;dim=1在每行上获取最大值以及最
        # 大值在这一行中的索引位置
        total += labels.size(0)  # 获取每批数据这个张量索引0上的大小，即每批数据的label数
        correct += (predicted == labels).sum()

        # 输出错误图片
        # result = (predicted != labels).numpy()
        # print(result)
        # for index in np.where(result == 1)[0]:
        #     print(index)
        #     imshow(images[index])

    print('Accuracy of the network on the test images:%d %%' %
          (100 * correct / total))


def practice(net, netName):
    '''
    开始训练
    '''
    data_loader = torch.utils.data.DataLoader(
        img_data, batch_size=100, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(20):
        for step, (x, y) in enumerate(data_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = net(b_x)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('this is epoch:', epoch, 'the step is', step,
                  'loss function is:', loss.data[0])
        testModel(net)
        torch.save(net, '/home/nifer/VSCode/Pytorch_Go/' + netName + '.pkl')


if __name__ == '__main__':

    practice(load_Mynet('lenet5_p'), 'lenet5_p')
    testModel(load_Mynet('lenet5_p'))
    # net = load_Mynet('lenet5')

    # data_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=100, shuffle=True)
    # img = None
    # for data in data_loader:
    #     images, labels = data  # 每次遍历得到的是数据集批训练个数的集合
    #     img = images
    #     outputs = net(Variable(images))  # 利用现有参数得到的输出，这里的output得到的是100×10矩阵
    #     _, predicted = torch.max(outputs.data, 1)
    #     break
    # print(img[0].size())
    # imshow(img[0])
    # img = img[0].view(1, 3, 100, 100)

    # v, i = torch.max(net(Variable(img)), 1)
    # result = test_data.class_to_idx
    # i = i.data.numpy()[0]
    # # print(result)
    # # print(i)
    # print('This Picture is:',
    #       list(result.keys())[list(result.values()).index(i)])
