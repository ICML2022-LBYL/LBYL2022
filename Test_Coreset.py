from __future__ import print_function
import warnings

warnings.simplefilter("ignore", UserWarning)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import pickle
import copy
import easydict
import random

cwd = os.getcwd()
sys.path.append(cwd + '/../')
import models
from models import Cifar100_ResNet
from models import ImageNet_ResNet
from models import LeNet_300_100
from torchvision import datasets, transforms
from torch.autograd import Variable
from LBYL import Decompose
from torchsummaryX import summary
import matplotlib.pyplot as plt


from typing import Callable, Tuple, Union
import sys

import torch
import numpy as np

cwd = os.getcwd()
sys.path.append(cwd + '/../')


class Coreset:
    def __init__(self, points, weights, activation_function: Callable, upper_bound: int = 1):
        assert points.shape[0] == weights.shape[0]

        self.__points = points.cpu()
        self.__weights = weights.cpu()
        self.__activation = activation_function
        self.__beta = upper_bound
        self.__sensitivity = None
        self.indices = None

    # @property
    def sensitivity(self):
        if self.__sensitivity is None:
            points_norm = self.__points.norm(dim=1)
            assert points_norm.shape[0] == self.__points.shape[0]
            weights = torch.abs(self.__weights).max(dim=1)[0]  # max returns (values, indices)
            assert weights.shape[0] == self.__points.shape[0]

            self.__sensitivity = weights * torch.abs(self.__activation(self.__beta * points_norm))
            self.__sensitivity /= self.__sensitivity.sum()

        return self.__sensitivity

    def compute_coreset(self, coreset_size):
        assert coreset_size <= self.__points.shape[0]
        prob = np.array(self.sensitivity())#  .cpu().numpy()
        prob /= prob.sum()
        points = self.__points
        indices = set()
        idxs = []


        cnt = 0
        while len(indices) < coreset_size:
            i = np.random.choice(a=points.shape[0], size=1, p=prob).tolist()[0]
            idxs.append(i)
            indices.add(i)
            cnt += 1

        hist = np.histogram(idxs, bins=range(points.shape[0] + 1))[0].flatten()
        idxs = np.nonzero(hist)[0]
        self.indices = idxs
        coreset = points[idxs, :]

        weights = (self.__weights[idxs].t() * torch.tensor(hist[idxs]).float()).t()
        weights = (weights.t() / (torch.tensor(prob[idxs]) * cnt)).t()

        return coreset, weights


def compress_fc_layer(layer1: Tuple[torch.tensor, torch.tensor],
                      layer2: Tuple[torch.tensor, torch.tensor],
                      compressed_size,
                      activation: Callable,
                      upper_bound,
                      device,
                      compression_type):
    num_neurons = layer1[1].shape[0]
    if compression_type == "Coreset":
        points = np.concatenate(
            (layer1[0].cpu().numpy(), layer1[1].view(num_neurons, 1).cpu().numpy()),
            axis=1) # weights, bias
        points = torch.tensor(points)
        weights = layer2[0].t()
        coreset = Coreset(points=points, weights=weights, activation_function=activation, upper_bound=upper_bound)
        points, weights = coreset.compute_coreset(compressed_size)
        indices = coreset.indices
        layer1 = (points[:, :-1].to(device), points[:, 1].to(device))
        weights = weights.t()
        layer2 = (weights.to(device), layer2[1].to(device))
    elif compression_type == "Uniform":
        indices = np.random.choice(num_neurons, size=compressed_size, replace=False)
        layer1 = (layer1[0][indices, :], layer1[1][indices])
        layer2 = (layer2[0][:, indices], layer2[1])
    elif compression_type == "Top-K":
        indices = torch.topk(torch.norm(layer1[0], dim=1), k=compressed_size)[1]
        layer1 = (layer1[0][indices, :], layer1[1][indices])
        layer2 = (layer2[0][:, indices], layer2[1])
    else:
        sys.exit("There is not a compression type: {}".format(compression_type))

    return layer1, layer2, indices



def save_state(model, acc):
    print('==> Saving model ...')
    state = {
        'acc': acc,
        'state_dict': model.state_dict(),
    }
    for key in state['state_dict'].keys():
        if 'module' in key:
            print(key)
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)

    if args.arch == 'LeNet_300_100':
        model_filename = '.'.join([args.arch,
                                   args.dataset,
                                   'Coreset',
                                   str(args.pruning_ratio),
                                   'pth.tar'])

    torch.save(state, os.path.join('saved_models/', model_filename))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))
    return


def test(epoch, evaluate=False):
    global best_acc, best_acc_list
    global best_epoch
    test_loss = 0
    correct = 0

    model.eval()

    with torch.no_grad():

        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Accuracy: {:.2f}%'.format(acc, ))

    best_acc_list.append(best_acc)

    # save the model
    save_state(model, best_acc)

    return


def weight_init(model, decomposed_weight_list):
    for layer in model.state_dict():
        decomposed_weight = decomposed_weight_list.pop(0)
        model.state_dict()[layer].copy_(decomposed_weight)

    return model


if __name__ == '__main__':

    ### settings
    parser = argparse.ArgumentParser(description='Coreset Implementation on LeNet-300-100')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
            help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
            help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
            metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--arch', action='store', default='LeNet_300_100',
            help='network structure: LeNet_300_100')
    parser.add_argument('--pretrained', action='store', default='./saved_models/LeNet_300_100.original.pth.tar',
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True, # False
            help='whether to run evaluation')
    parser.add_argument('--retrain', action='store_true', default=True, # False
            help='whether to retrain')
    parser.add_argument('--dataset', action='store', default='fashionMNIST',
            help='dataset: fashionMNIST')
    parser.add_argument('--pruning-ratio', type=float, default=0.5,
            help='pruning ratio : (default: 0.1)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
            help='pin memory : (default: True)')
    parser.add_argument('--data-loader-workers', type=int, default=4,
            help='data loader works: (default: 4)')
    args = parser.parse_args()
    best_acc_list = []

    # args = easydict.EasyDict()
    # args.batch_size = 128  # 128
    # args.test_batch_size = 256
    # args.epochs = 200
    # args.lr = 0.0001  # 0.1
    # args.momentum = 0.9
    # args.weight_decay = 5e-4
    # args.no_cuda = False  # action = 'strore_true', default = false
    # args.seed = 1
    # args.log_interval = 100
    # args.arch = 'LeNet_300_100'
    # args.pretrained = './saved_models/LeNet_300_100.original.pth.tar'
    # args.evaluate = True  # False # action = 'store_true', defalut = True
    # args.retrain = True  # action='store_true', default=False
    # args.model_type = 'OURS'  # prune | merge | OURS
    # args.dataset = 'fashionMNIST'  # fashionMNIST
    # args.pruning_ratio = 0.5
    # args.gpu = '0'
    # args.pin_memory = True
    # args.data_loader_workers = 4  # 8
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # args.schedule = [100, 200]
    # # args = parser.parse_args()
    # print(args)
    # best_acc_list = []

    # check options
    if not (args.arch in ['LeNet_300_100']):
        print('ERROR: specified arch is not suppported')
        exit()

    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU

    # load data
    if args.dataset == 'fashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        num_classes = 10


    cfg = None

    # make cfg
    if args.retrain:

        if 'LeNet_300_100':
            cfg = [300, 100]
            for i in range(len(cfg)):
                cfg[i] = round(cfg[i] * (1 - args.pruning_ratio))
            # cfg = [60,20] # cfg

    # generate the model
    if args.arch == 'LeNet_300_100':
        model = LeNet_300_100.LeNet_300_100(bias_flag=True, cfg=cfg)
    else:
        pass

    if args.cuda:
        model.cuda()

    # pretrain
    best_acc = 0.0
    best_epoch = 0

    if args.pretrained:
        if args.arch == 'LeNet_300_100':
            pretrained_model = torch.load(args.pretrained)
        best_epoch = 0

    print(model)

    for layer in pretrained_model['state_dict']:

        if layer in ['ip1.weight', 'ip2.weight']:

            if layer in 'ip1.weight':
                ip1_weight = pretrained_model['state_dict']['ip1.weight']
                ip1_bias = pretrained_model['state_dict']['ip1.bias']
                ip2_weight = pretrained_model['state_dict']['ip2.weight']
                ip2_bias = pretrained_model['state_dict']['ip2.bias']

                layer1, _layer2, indices = compress_fc_layer(layer1=[ip1_weight, ip1_bias], layer2=[ip2_weight, ip2_bias],
                                                            compressed_size=cfg[0], activation=torch.nn.ReLU(), upper_bound=1,
                                                            device='cuda',
                                                            compression_type='Coreset')
                pretrained_model['state_dict']['ip1.weight'] = layer1[0]
                pretrained_model['state_dict']['ip1.bias'] = layer1[1]


            elif layer in 'ip2.weight':
                # ip2_weight = pretrained_model['state_dict']['ip2.weight']
                # ip2_bias = pretrained_model['state_dict']['ip2.bias']
                ip2_weight = _layer2[0]
                ip2_bias =_layer2[1]

                ip3_weight = pretrained_model['state_dict']['ip3.weight']
                ip3_bias = pretrained_model['state_dict']['ip3.bias']

                layer2, layer3, indices = compress_fc_layer(layer1=[ip2_weight, ip2_bias], layer2=[ip3_weight, ip3_bias],
                                                            compressed_size=cfg[1], activation=torch.nn.ReLU(), upper_bound=1,
                                                            device='cuda',
                                                            compression_type='Coreset')

                pretrained_model['state_dict']['ip2.weight'] = layer2[0]
                pretrained_model['state_dict']['ip2.bias'] = layer2[1]
                pretrained_model['state_dict']['ip3.weight'] = layer3[0]
                pretrained_model['state_dict']['ip3.bias'] = layer3[1]

    for layer in pretrained_model['state_dict'].values():
        print(layer.shape)

    model.load_state_dict(pretrained_model['state_dict'])
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total parameter number:', params, '\n')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        test(0, evaluate=True)
        # summary(model, torch.zeros((1, 3, 256, 256)).cuda())
    exit()
