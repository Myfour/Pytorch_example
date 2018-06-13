import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms
# import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决图片截断的问题
# 训练集
img_data = torchvision.datasets.ImageFolder(
    '/home/nifer/Desktop/TheFinalDoc/图片集',
    transform=transforms.Compose([
        transforms.Resize((500, 500)),  # 注意使用方式,若只定义一个int值,则是将一个较小维度变为这个值
        # transforms.CenterCrop(
        #    500),  # 将图片从中心裁剪,参数若为一个int类型则裁剪成长度为int类型值的正方形,若为(h,w)则为相应矩形
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
# 测试集
test_data = torchvision.datasets.ImageFolder(
    '/home/nifer/Desktop/TheFinalDoc/测试集',
    transform=transforms.Compose([
        transforms.Resize((500, 500)),  # 注意使用方式,若只定义一个int值,则是将一个较小维度变为这个值
        # transforms.CenterCrop(
        #    500),  # 将图片从中心裁剪,参数若为一个int类型则裁剪成长度为int类型值的正方形,若为(h,w)则为相应矩形
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

print(len(img_data))  # 读取到的总图片数
data_loader = torch.utils.data.DataLoader(
    img_data, batch_size=20, shuffle=True)
print(len(data_loader))  # 批处理的次数,每次10张图片


#  LeNet-5模型
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.c1 = nn.Sequential(  # (3,500,500)
            nn.Conv2d(3, 6, 5, 1, 2),  # (6,500,500)
            nn.ReLU())
        self.s2 = nn.MaxPool2d(2)  # (6,250,250)
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  # (16,246,246)
            nn.ReLU())
        self.s4 = nn.MaxPool2d(2)  # (16,123,123)
        self.c5 = nn.Linear(16 * 123 * 123, 120)
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


lenet5 = LeNet_5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet5.parameters(), lr=0.001)
print(lenet5)
'''
开始训练
'''
for epoch in range(1):
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = lenet5(b_x)
        loss = criterion(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('this is epoch:', epoch, 'the step is', step)
        print('loss function is:', loss.data)

torch.save(lenet5, '/home/nifer/VSCode/Pytorch_Go/lenet5.pkl')
'''
测试训练的结果
'''
data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=True)
correct = 0
total = 0
for data in data_loader:
    images, labels = data  # 每次遍历得到的是数据集批训练个数的集合
    outputs = lenet5(Variable(images))  # 利用现有参数得到的输出，这里的output得到的是10×4的矩阵
    _, predicted = torch.max(outputs.data,
                             1)  # torch.max()获取输入张量的最大值，加入dim参数后，
    # 返回的是在给定维度上的最大值;dim=1在每行上获取最大值以及最
    # 大值在这一行中的索引位置
    total += labels.size(0)  # 获取这个张量索引0上的大小
    correct += (predicted == labels).sum()
print('Accuracy of the network on the test images:%d %%' %
      (100 * correct / total))
