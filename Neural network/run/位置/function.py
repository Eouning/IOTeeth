import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import os, random, shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from itertools import cycle
from torchvision import datasets
from torch.autograd import Variable
import torchvision
from torchvision.transforms import transforms
from visdom import Visdom
from torch.utils.data import Dataset, DataLoader, TensorDataset,ConcatDataset

def Make_loader(data_train,data_label,batch_size=200, shuffle=False):
    """
    介绍：制作DataLoader \n
    传入变量: \n
            data_train:传入的图片集合 \n
            data_label:传入的标签集合 \n
            batch_size:每一组的数据数量，用于划分训练集 \n
            shuffle:是否对传入的数据及进行随机打乱 \n
    传出变量： \n
            data_loader:传出制作好的DataLoader \n
    """
    
    dataset = TensorDataset(data_train,data_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#shuffle是是否打乱数据集，可自行设置

    return data_loader

def get_loader(fileDir,tarDir,file_label_train,file_label_val,batch_size=200,shuffle=False):
    """
    介绍：从文件中读取数据并制作数据集 \n
    传入变量: \n
            fileDir:训练图片文件夹路径 \n
            tarDir:验证图片文件夹路径 \n
            file_label_train:训练标签文件夹路径 \n
            file_label_val:验证标签文件夹路径 \n
            batch_size:每一组的数据数量，用于划分训练集 \n
            shuffle:是否对传入的数据及进行随机打乱 \n
    传出变量： \n
            train_loader:传出制作好的训练数据集 \n
            test_loader:传出制作好的验证数据集 \n
    """
    
    transf = torchvision.transforms.ToTensor()  # 实例化类
    device=torch.device('cuda:0')  #在GPU上运行
    
    img_list=[]
    label_list=[]
    img_list_val=[]
    label_list_val=[]
    name_list = os.listdir(fileDir)
    name_val_list = os.listdir(tarDir)
    for name in name_list:
        f=open(file_label_train+'/'+name[:-4]+'.txt')
        text=f.read()
        if text=='':
            print('训练集 '+name+' 的标签为空')
            f.close()
        else:
            im=cv2.imread(fileDir+'/'+name)
            #lenth=max([im.shape[0],im.shape[1]])
            label_list.append(int(text))
            im=cv2.resize(im,(128,128), interpolation = cv2.INTER_CUBIC)
            img_list.append(transf(im))
            f.close

    for name in name_val_list:
        f=open(file_label_val+'/'+name[:-4]+'.txt')
        text=f.read()
        if text=='':
            print('训练集 '+name+' 的标签为空')
            f.close()
        else:
            im=cv2.imread(tarDir+'/'+name)
            #lenth=max([im.shape[0],im.shape[1]])
            label_list_val.append(int(text))
            im=cv2.resize(im,(128,128), interpolation = cv2.INTER_CUBIC)
            img_list_val.append(transf(im))
            f.close


    data=torch.stack(img_list)
    data_val=torch.stack(img_list_val)
    target=torch.tensor(label_list)
    target_val=torch.tensor(label_list_val)

    data=data.to(torch.float32)
    data_val=data_val.to(torch.float32)
    target=target.to(torch.long)
    target_val=target_val.to(torch.long)
                        
    print('')
    print('训练图片集规模：',data.shape)
    print('训练标签集规模：',target.shape)
    print('验证图片集规模：',data_val.shape)
    print('验证标签集规模：',target_val.shape)
    
    train_loader,test_loader=Make_loader(data,target,batch_size,shuffle) , Make_loader(data_val,target_val,batch_size,False)
    return train_loader,test_loader  

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)

class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class Small_8(nn.Module):
    def __init__(self):
        super(Small, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 8)  #分类数在这改
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 4)  #分类数在这改
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
def model_save(model,path):
    """
    介绍：保存模型为权重文件 \n
    传入变量: \n
            model:要保存权重的模型 \n
            path:要保存的路径 \n
    传出变量： \n
            无 \n
    """
    torch.save(model.state_dict(),path)

def train(model,train_loader,test_loader,learn_rate,epochs,batch_size,save_dir):
    """
    介绍：网路训练函数。训练可中断，同时保留训练效果。本函数使用了visdom以可视化数据。使用命令 python -m visdom.server 打开visdom \n
         可视化训练效果网址：http://localhost:8097 \n
    传入变量: \n
            model:要训练的模型 \n
            train_loader:训练数据集 \n
            test_loader:验证数据集 \n
            tarDir:验证图片文件夹路径 \n
            learn_rate:学习率 \n
            epochs:训练轮数 \n
            batch_size:每一组的数据数量，用于划分训练集 \n
            save_dir:存储模型的文件夹 \n
    传出变量： \n
            无 \n
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device=torch.device('cuda:0')  #在GPU上运行

    optimizer=torch.optim.Adam(model.parameters(), learn_rate)
    loss_fun=nn.CrossEntropyLoss().to(device)
    
    if save_dir[-1]!="/":
        save_dir=save_dir+'/'

    """
    #使用命令 python -m visdom.server 打开visdom
    #网址：http://localhost:8097
    viz=Visdom()
    viz.line([0.],[0.],win='train_loss',opts=dict(title='train loss'))
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='test loss&acc',
                                               legend=['loss','acc']))
    """

    best=0
    step_Tr=0
    step_Te=0

    for epoch in range(epochs):
    
        for batch_idx,(data,target) in enumerate(train_loader):
        
            step_Tr+=1
        
            data,target=data.to(device),target.to(device)
        
            logits=model(data)
            loss=loss_fun(logits,target)
        
            #viz.line([loss.item()],[step_Tr],win='train_loss',update='append')
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if batch_idx %100.==0:
                print('训练轮数: {} [{}/{} ({:.0f}%) ]\tLoss: {:.9f}'.format(
                    epoch,batch_idx*len(data),len(train_loader.dataset),
                        100*(batch_idx*len(data))/len(train_loader.dataset),loss.item()))
        test_loss=0
        correct=0
        step_Te+=1
        
        for data,target in test_loader:
        
            data,target=data.to(device),target.to(device)
            
            logits=model(data)
            test_loss+=loss_fun(logits,target).item()*len(data)
    
            pred=logits.data.max(1)[1]
            correct +=pred.eq(target.data).sum()
                    
        test_loss/=len(test_loader.dataset)

        #viz.line([[test_loss,correct.cpu()/len(test_loader.dataset)]],
                        #[step_Te],win='test',update='append')
    
        print('\n结果验证: 平均损失: {:.9f},准确率: {}/{} ({:.0f}%)\n'.format(
            test_loss,correct,len(test_loader.dataset), 
        100.*correct/len(test_loader.dataset)))
        
        if 100.*correct/len(test_loader.dataset)>best:
            best=100.*correct/len(test_loader.dataset)
            model_save(model,save_dir+str(batch_size)+"|"+str(epoch)+'|'+str(float(100.*correct/len(test_loader.dataset)))+'.pth')
            print("<--------->")
            print("第"+str(epoch)+"轮保存成功")
            print("<--------->")

    print('训练结束！')

def model_load(model,path):
    """
    介绍：将权重文件加载到模型上 \n
    传入变量: \n
            model:要加载权重的模型 \n
            path:要加载的权重的路径 \n
    传出变量： \n
            无 \n
    """
    model.load_state_dict(torch.load(path))

def Merge_loader(Loader_1, Loader_2, batch_size, shuffle):
    """
    介绍：本函数可混合两个DataLoader \n
    传入变量: \n
            Loader_1:混合对象1 \n
            Loader_2:混合对象2 \n
            batch_size:每个batch的size \n
            shuffle:是否随即打乱数据 \n
    传出变量： \n
            merged_dataloader:混合后的DataLoader \n
    """
    
    # 将两个 DataLoader 的数据集合并为一个 ConcatDataset 对象
    dataset_1=Loader_1.dataset
    
    img_list=[]
    label_list=[]
    for x in range(len(Loader_2.dataset)):
        img_list.append(Loader_2.dataset[x][0])
        label_list.append(Loader_2.dataset[x][1])
        
    data=torch.stack(img_list)
    target=torch.tensor(label_list)
    data=data.to(torch.float32)
    target=target.to(torch.long)
    
    dataset_2 = TensorDataset(data,target)
        
    concat_dataset = ConcatDataset([dataset_1,dataset_2])
    
    # 创建合并后的 DataLoader 对象
    merged_dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=shuffle)

    return merged_dataloader

def testimg(img,model):
    """
    介绍：验证模型对单张图片的识别情况 \n
    传入变量: \n
            img:需识别的图片 \n
            model:要验证的模型 \n
    传出变量： \n
            num:验证结果
    """ 
    Resize_num=128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transf = torchvision.transforms.ToTensor()  # 实例化类
    img_list=[]
    im=cv2.resize(img,(Resize_num, Resize_num), interpolation = cv2.INTER_CUBIC)
    img_list.append(transf(im))
    data=torch.stack(img_list)
    data=data.to(torch.float32)
    with torch.no_grad():
        logits=model(data.to(device))
    num=int(logits.data.max(1)[1].tolist()[0])
    return num

def run(img,model_position):
    """
    介绍：输出模型的识别结果 \n
    传入变量: \n
            img:需识别的图片 \n
            model_position:所用模型的地址 \n
    传出变量： \n
            num:验证结果 \n
                0:无口腔 \n
                1:down下牙上侧 \n
                2:front \n 
                3:up \n 
    """ 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    let=Small().to(device)
    model_load(let,model_position)
    return testimg(img,let)

def list_run(img_list,model_position):
    """
    介绍：输出针对图像列表的模型的识别结果，可以加快图像列表的识别效率 \n
    传入变量: \n
            img_list:需识别的图片列表 \n
            model_position:所用模型的地址 \n
    传出变量： \n
            num:验证结果 \n
                0:无口腔 \n
                1:down \n
                2:front \n 
                3:up \n 
    """ 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    let=Small().to(device)
    model_load(let,model_position)
    result_list=[]
    for img in img_list:
        result_list.append(testimg(img,let))
    return result_list