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

    if args.arch == 'ResNet':
        model_filename = '.'.join([args.arch,
                                   str(args.depth_wide),
                                   args.dataset,
                                   args.model_type,
                                   args.criterion,
                                   str(args.pruning_ratio),
                                   'pth.tar'])
    else:
        model_filename = '.'.join([args.arch,
                                   args.dataset,
                                   args.model_type,
                                   args.criterion,
                                   str(args.pruning_ratio),
                                   'pth.tar'])

    torch.save(state, os.path.join('saved_models/', model_filename))

def adjust_learning_rate(optimizer, epoch, gammas, schedule):

    lr = args.lr
    for (gamma, step) in zip (gammas, schedule):
        if(epoch>= step) and (args.epochs * 3 //4 >= epoch):
            lr = lr * gamma
        elif(epoch>= step) and (args.epochs * 3 //4 < epoch):
            lr = lr * gamma * gamma
        else:
            break
    print('learning rate : ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return

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
    print('Accuracy: {:.2f}%'.format(acc,))

    best_acc_list.append(best_acc)

    # save the model
    save_state(model, best_acc)

    return



def weight_init(model, decomposed_weight_list):
    for layer in model.state_dict():
        decomposed_weight = decomposed_weight_list.pop(0)
        # print(layer, decomposed_weight.shape, model.state_dict()[layer].shape )
        model.state_dict()[layer].copy_(decomposed_weight)

    return model


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='LBLY Example')
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
            help='network structure: LeNet_300_100 | VGG16 | ResNet50 | ResNet34 | ResNet101')
    parser.add_argument('--pretrained', action='store', default='./saved_models/LeNet_300_100.original.pth.tar',
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True, # False
            help='whether to run evaluation')
    parser.add_argument('--retrain', action='store_true', default=True, # False
            help='whether to retrain')
    parser.add_argument('--model-type', action='store', default='OURS', # merge
            help='model type: prune | merge | OURS')
    parser.add_argument('--dataset', action='store', default='cifar10',
            help='dataset: fashionMNIST | cifar10 | cifar100 | ImageNet')
    parser.add_argument('--criterion', action='store', default='l2-norm',
            help='criterion : l1-norm | l2-norm | l2-GM | random')
    parser.add_argument('--threshold', type=float, default=0.1,
            help='threshold (default: 0.1)')
    parser.add_argument('--lamda-1', type=float, default=0.8,
            help='lamda (default: 0.85)')
    parser.add_argument('--lamda-2', type=float, default=0.8,
            help='lamda (default: 0.1)')
    parser.add_argument('--pruning-ratio', type=float, default=0.1,
            help='pruning ratio : (default: 0.1)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
            help='pin memory : (default: True)')
    parser.add_argument('--data-loader-workers', type=int, default=4,
            help='data loader works: (default: 4)')
    args = parser.parse_args()

    # args = easydict.EasyDict()
    # args.batch_size = 128  # 128
    # args.test_batch_size = 256
    # args.epochs = 20
    # args.lr = 0.00001  # 0.1
    # args.momentum = 0.9
    # args.weight_decay = 5e-4
    # args.no_cuda = False  # action = 'strore_true', default = false
    # args.seed = 1
    # args.log_interval = 100
    # args.arch =  'LeNet_300_100'  # ResNet50  , VGG16, ResNet50, ResNet34, ResNet101
    # args.pretrained = './saved_models/LeNet_300_100.original.pth.tar' # './saved_models/ResNet.cifar100.original.50.pth.tar' #'./saved_models/VGG.cifar10.original.pth.tar'
    # args.evaluate = True  # False # action = 'store_true', defalut = True
    # args.retrain = True  # action='store_true', default=False
    # args.model_type = 'OURS'  #  prune | merge | OURS | coreset
    # args.dataset = 'fashionMNIST'  # fashionMNIST |cifar10 | cifar100 | ImageNet
    # args.criterion = 'l2-norm'  #  l1-norm | l2-norm | l2-GM | random
    # args.threshold = 0.15
    # args.lamda_1 =4e-6
    # args.lamda_2 = 0.02
    # args.pruning_ratio = 0.1
    # args.gpu = '0'
    # args.pin_memory = True
    # args.data_loader_workers = 4 # 8
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # args.schedule = [5, 10, 15]
    # args.gammas = [0.1,0.1,0.1]
    # args = parser.parse_args()
    print(args)
    best_acc_list = []

    # check options
    if not (args.model_type in ['merge', 'prune', 'coreset','OURS']):
        print('ERROR: Please choose the correct model type')
        exit()
    if not (args.arch in ['LeNet_300_100','VGG16', 'ResNet50', 'ResNet34', 'ResNet101']):
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
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        num_classes = 10


    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=2)
        bn_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=2)

        num_classes = 10

    elif args.dataset == 'cifar100':
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR100("./data", train=False, transform=valid_transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                       num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                       num_workers=2)
        num_classes = 100

    else: # ImageNet
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


        train_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_set = datasets.ImageFolder(os.path.join("C:/Users/USER001/ILSVRC2012_img_validation"), transform=valid_transform)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.data_loader_workers, pin_memory=args.pin_memory)


    cfg = None

    # make cfg
    if args.retrain:
        if args.arch == 'VGG16':
            if args.dataset == 'cifar10':
                # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
                if args.pruning_ratio == 0.1:
                    cfg = [57, 57, 'M', 115, 115, 'M', 230, 230, 230, 'M', 460, 460, 460, 'M', 460, 460, 460]
                elif args.pruning_ratio == 0.2:
                    cfg = [51, 51, 'M', 102, 102, 'M', 204, 204, 204, 'M', 409, 409, 409, 'M', 409, 409, 409]
                elif args.pruning_ratio == 0.3:
                    cfg = [44, 44, 'M', 89, 89, 'M', 179, 179, 179, 'M', 358, 358, 358, 'M', 358, 358, 358]
                elif args.pruning_ratio == 0.4:
                    cfg = [38, 38, 'M', 76, 76, 'M', 153, 153, 153, 'M', 307, 307, 307, 'M', 307, 307, 307]
                elif args.pruning_ratio == 0.5:
                    cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256]
                temp_cfg = list(filter(('M').__ne__, cfg))

        elif 'ResNet' in args.arch:
            cfg = [64, 128, 256, 512]
            for i in range(len(cfg)):
                cfg[i] = int(cfg[i] * (1 - args.pruning_ratio))
            temp_cfg = cfg

        elif 'LeNet_300_100':
            cfg = [300, 100]
            for i in range(len(cfg)):
                cfg[i] = round(cfg[i] * (1 - args.pruning_ratio))
            temp_cfg = cfg

    # generate the model
    if args.arch == 'LeNet_300_100':
        model = LeNet_300_100.LeNet_300_100(bias_flag=True, cfg=cfg)
    elif args.arch == 'VGG16':
        model = models.VGG(num_classes, cfg=cfg)
    elif args.arch =='ResNet50':
        model = Cifar100_ResNet.resnet50(cfg = cfg)
    elif args.arch == 'ResNet34':
        model = models.ImageNet_ResNet.resnet34(pretrained= False, cfg=cfg)
    elif args.arch == 'ResNet101':
        model = models.ImageNet_ResNet.resnet101(pretrained = False, cfg= cfg)
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
        elif args.arch == 'VGG16' or args.arch == 'ResNet50':
            pretrained_model = torch.load(args.pretrained)
        elif args.arch == 'ResNet34':
            pretrained_model = models.ImageNet_ResNet.resnet34(pretrained=True)
            pretrained_model.cuda()
        elif args.arch =='ResNet101':
            pretrained_model = models.ImageNet_ResNet.resnet101(pretrained=True)
            pretrained_model.cuda()
        best_epoch = 0


    # weight initialization
    if args.retrain:
        if args.arch == 'LeNet_300_100':
            decomposed_list = Decompose(args.arch, pretrained_model['state_dict'],
                                                                args.criterion,
                                                                args.threshold,
                                                                args.lamda_1, args.lamda_2,
                                                                args.model_type, temp_cfg, args.cuda).main()
            model = weight_init(model, decomposed_list)

        elif args.arch == 'VGG16':
            decomposed_list = Decompose(args.arch, pretrained_model['state_dict'],
                                                                args.criterion,
                                                                args.threshold,
                                                                args.lamda_1, args.lamda_2,
                                                                args.model_type, temp_cfg, args.cuda).main()
            model = weight_init(model, decomposed_list)
        elif args.arch == 'ResNet50':
            decomposed_list = Decompose(args.arch, pretrained_model,
                                                                args.criterion,
                                                                args.threshold,
                                                                args.lamda_1, args.lamda_2,
                                                                args.model_type, temp_cfg, args.cuda).main()
            model = weight_init(model, decomposed_list)

        elif args.arch == 'ResNet34' or args.arch == 'ResNet101':
            decomposed_list = Decompose(args.arch, pretrained_model.state_dict(),
                                        args.criterion,
                                        args.threshold,
                                        args.lamda_1, args.lamda_2,
                                        args.model_type, temp_cfg, args.cuda).main()
            model = weight_init(model, decomposed_list)

    print(model)
    # print the number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total parameter number:', params, '\n')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    

    if args.evaluate:
        test(0, evaluate=True)
#         summary(model, torch.zeros((1, 3, 256, 256)).cuda())
        # exit()

    # for epoch in range(1, args.epochs + 1):
    #     adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
    #     train(epoch)
    #     test(epoch)