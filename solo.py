#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2023-05-08 17:29'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset,DataLoader
from InjectNoise import generate_noisy_label
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv,os

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义功能函数
def look_differ_list(list1, list2):
    '''
    list1: List type
    list2: List type
    return diff_count,percent_diff
    diff_count: The number of elements that differ between two lists
    percent_diff: Percentage of the length of the list with distinct elements
    Note that: list1 and list2 are required to have the same length
    '''

    diff_count = 0
    for x, y in zip(list1, list2):
        if x != y:
            diff_count += 1

    percent_diff = diff_count / len(list1) * 100

    return diff_count, percent_diff


# 定义数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
print('####################################',len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
print('@@@@@@@@@@@@@@@@@@@@@@',len(train_loader),len(train_loader_all))
# 定义模型和优化器
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义损失函数和阈值
criterion = nn.CrossEntropyLoss()

epoch_loss = []
epoch_grad = []
# 训练模型
for epoch in range(1):
    model.train()
    loss_noise_img = []
    loss_noise_label = []
    grad_noise_img = []
    grad_noise_label = []
    remain_img =[]
    remain_label = []
    loss_god_label =[]
    grad_god_label = []
    loss_find_index = []
    grad_find_index = []

    intersection_noise_img = []
    intersection_noise_label = []
    intersection_god_label = []

    union_noise_img = []
    union_noise_label = []
    union_god_label = []
    for batch_idx, (img, label) in enumerate(tqdm(train_loader)):
        # print('lllllllllllll',len(label))
        optimizer.zero_grad()
        logit = model(img)
        # noisy_labels, _, _ = generate_noisy_label(label, 10, noise_type="symmetry", noise_ratio=0.8)
        noisy_labels, _, _ = generate_noisy_label(label, 10, noise_type="asymmetry", noise_ratio=0.9)
        loss = criterion(logit, noisy_labels)
        sample_loss_list=[]
        sample_grad_list=[]

        # print(img.shape[0]) #128

        for j in range(img.shape[0]):
            sample_loss = nn.functional.cross_entropy(logit[j].unsqueeze(0), label[j].unsqueeze(0))
            sample_loss_list.append((j,sample_loss))

            sample_loss.backward(retain_graph=True)
            sample_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    sample_grads.append(p.grad.view(-1))
            sample_grads = torch.cat(sample_grads)
            sample_grad_norm = torch.norm(sample_grads, p=2)
            sample_grad_list.append((j,sample_grad_norm))


        loss_noise_index = max(sample_loss_list,key= lambda x : x[1])[0]
        # print(loss_noise_index)
        # print(sample_loss_list)

        loss_find_index.append(loss_noise_index)

        grad_noise_index = min(sample_grad_list,key=lambda  x:x[1])[0]
        grad_find_index.append(grad_noise_index)
        grad_noise_img.append(img[grad_noise_index])
        grad_noise_label.append(noisy_labels[grad_noise_index])
        grad_god_label.append(label[grad_noise_index])

        if loss_noise_index == grad_noise_index:
            # print(loss_noise_index)
            intersection_noise_img.append(img[loss_noise_index])
            intersection_noise_label.append(noisy_labels[loss_noise_index])
            intersection_god_label.append(label[loss_noise_index])



        loss.backward()
        optimizer.step()



    intersection_noise_label = [k.item() for k in intersection_noise_label]
    intersection_god_label = [k.item() for k in intersection_god_label]
    intersection_differ, intersection_percent = look_differ_list(intersection_god_label,intersection_noise_label)

    '''sum is used to verify the selected noisy number and remain number'''
    sum =0
    for batch_i, (img_all, label_all) in enumerate(train_loader_all):
        for (x,y) in tqdm(zip(img_all,label_all)):
            count = 0
            for k in intersection_noise_img:
                flag = torch.equal(x,k)
                if flag:
                    count = count +1
            if count==0:
                sum=sum+1
                remain_img.append(x)
                remain_label.append(y)

        # print(']]]]]]]]]]]]]]]]]]',len(img_all))


    #print(len(intersection_noise_label),sum,len(intersection_noise_img),len(remain_label),len(remain_img))


    #saving d_noisy
    d_noisy = TensorDataset( torch.tensor([item.cpu().detach().numpy() for item in intersection_noise_img]),torch.tensor(intersection_noise_label))
    d_clean = TensorDataset(torch.tensor([item.cpu().detach().numpy() for item in remain_img]),torch.tensor(remain_label))

    torch.save(d_noisy, 'd_asym0.9_noisy.pt')
    print('d_noisy dataset saved!')
    torch.save(d_clean, 'd_asym0.9_clean.pt')
    print('d_clean dataset saved!')

    # breakpoint()
    # loss_noisy_dataset = TensorDataset(torch.tensor(loss_noise_img),torch.tensor(loss_noise_label))
    # torch.save(loss_noisy_dataset, 'loss_noisy_dataset.pt')
    # print('loss_noisy_dataset saved!')
    #
    # grad_noise_dataset = TensorDataset(torch.tensor(grad_noise_img),torch.tensor(grad_noise_label))
    # torch.save(grad_noise_dataset, 'grad_noisy_dataset.pt')
    # print('grad_noisy_dataset saved!')


        # 将损失最大的样本当做噪声样本
        # max_loss_index = torch.argmax(sample_losses)
        # max_loss = sample_losses[max_loss_index].item()
        # if max_loss > threshold:
        #     noise_data.append(train_dataset[max_loss_index])
        # else:
        #     clean_data.append(train_dataset[max_loss_index])

# 将干净数据集和噪声数据集保存到文件
# torch.save(clean_data, 'clean_data.pt')
# torch.save(noise_data, 'noise_data.pt')

#输入数据集 每次挑出噪声数据 从数据集中删除该数据 将剩余的数据集再次送入 再次筛选
#已知噪声的索引 求剩余的数据索引
#需要确定数据集的迭代轮数，使用递归思想

