import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
