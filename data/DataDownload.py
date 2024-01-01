#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2023-02-25 17:32'

from torchvision import datasets

#downloading MNIST
trainset_mnist = datasets.MNIST(root='./data',train=True,download=True)
testset_mnist = datasets.MNIST(root='./data',train=False,download=True)

#downloading CIFAR10
trainset_cifar10 = datasets.CIFAR10(root='./data',train=True,download=True)
testset_cifar10 = datasets.CIFAR10(root='./data',train=False,download=True)

#downloading CIFAR100
trainset_cifar100 = datasets.CIFAR100(root='./data',train=True,download=True)
testset_cifar100 = datasets.CIFAR100(root='./data',train=False,download=True)

