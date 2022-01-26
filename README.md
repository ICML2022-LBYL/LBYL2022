# LBYL - code
This is the authors' implementation of the following paper: LeaveBeforeYouLeave: Training-Free Restoration of Pruned Neural Networks Without Fine-Tuning
<img src="https://github.com/ICML2022-LBYL/LBYL2022/blob/main/images/LBYL_figure_1.png" width="100%" height="100%">


# Contents
1. [Requirements](#1-Requirements)<br>
2. [Pre-trained models and Dataset](#2-Pre-trained-models-and-Dataset)<br>
3. [Our experimental setting(GPU and CPU)](#3-Our-experimental-setting)<br>

## 1 Requirements
Python environment & main libraries:
- python 3.7
- pytorch 1.7
- torchvision 0.8
- scikit-learn 0.23
- numpy 1.19
- scipy 1.5
- torchsummaryx 1.3.0


## 2 Pre-trained models and Dataset
We release the pretrained models for CIFAR-10 and CIFAR-100 in save_models directory and also use pretrained ResNet-34 and ResNet-101 on ImageNet, both of which are released by PyTorch. If you run the experiments for ImageNet, you should download the ImageNet(ILSVRC2012) validatation set.



### Arguments
*Required*:
* `--dataset`: Choose datset. *Option*: `fashionMNIST` or `cifar10` or `cifar100` or `ImageNet` 
* `--arch` : Choose architecture *Option*: `LeNet_300_100` on fashionMNIST or `VGG16` on cifar10 or `ResNet50` on cifar100  or `ResNet34` on ImageNet or `ResNet101` on ImageNet
* `--model-type`: Choose model type *Option*: `OURS` or `merge` or `prune`  or `coreset` for LeNet-300-100 and ResNet50 
* `--criterion` : Choose criterion *Option*: `l2-norm` or `l2-GM` or `l1-norm` or `random_1` or `random_2` or `random_3`
* `--lamda-1` : Choose lambda_1 
* `--lamda-2` : Choose lambda_2
* `--pruning-ratio` : Choose pruning ratio 


### LeNet-300-100 on FashionMINST
The following results can be reproduced with command:

    python main.py --arch LeNet_300_100 --pretrained ./saved_models/LeNet_300_100.original.pth.tar --model-type OURS --dataset fashionMNIST --criterion l2-norm --lamda-1 0.0 --lamda-2 0.3 --pruning-ratio 0.5
    python main.py --arch LeNet_300_100 --pretrained ./saved_models/LeNet_300_100.original.pth.tar --model-type OURS --dataset fashionMNIST --criterion l2-norm --lamda-1 0.0 --lamda-2 0.6 --pruning-ratio 0.6
    python main.py --arch LeNet_300_100 --pretrained ./saved_models/LeNet_300_100.original.pth.tar --model-type OURS --dataset fashionMNIST --criterion l2-norm --lamda-1 0.0 --lamda-2 0.3 --pruning-ratio 0.7
    python main.py --arch LeNet_300_100 --pretrained ./saved_models/LeNet_300_100.original.pth.tar --model-type OURS --dataset fashionMNIST --criterion l2-norm --lamda-1 0.0 --lamda-2 1e-06 --pruning-ratio 0.8
    
 Pruning Criterion : L2 - norm 
 
| Pruning   ratio | lamda2 | acc(Ours) | acc(NM) | acc(prune) |
|:---------------:|:------:|:---------:|:-------:|:----------:|
|       50%       | 0.3 |   88.83   |  87.86  |    87.86   |
|       60%       |   0.6 |   87.75   |  88.07  |    83.03   |
|       70%       |   0.3  |   83.92   |  83.27  |    71.21   |
|       80%       |   1e-06  |   78.05   |  77.11  |    63.9   |

We offer the implementation of Coreset in LeNet-300-100 on FashionMNIST. If you test the implementation of Coreset, run the below command. 

    python Test_Coreset.py --pruning-ratio 0.5
    
    
    

### VGG16 on CIFAR-10
The following results can be reproduced with command:

    python main.py --arch VGG16 --pretrained ./saved_models/VGG.cifar10.original.pth.tar --model-type OURS --criterion l2-norm --lamda-1 0.000006 --lamda-2 0.0001 --pruning-ratio 0.1
    python main.py --arch VGG16 --pretrained ./saved_models/VGG.cifar10.original.pth.tar --model-type OURS --criterion l2-norm --lamda-1 0.000004 --lamda-2 0.006 --pruning-ratio 0.2
    python main.py --arch VGG16 --pretrained ./saved_models/VGG.cifar10.original.pth.tar --model-type OURS --criterion l2-norm --lamda-1 0.000001 --lamda-2 0.01 --pruning-ratio 0.3
    python main.py --arch VGG16 --pretrained ./saved_models/VGG.cifar10.original.pth.tar --model-type OURS --criterion l2-norm --lamda-1 0.000002 --lamda-2 0.01 --pruning-ratio 0.4
    python main.py --arch VGG16 --pretrained ./saved_models/VGG.cifar10.original.pth.tar --model-type OURS --criterion l2-norm --lamda-1 0.00004 --lamda-2 0.0002 --pruning-ratio 0.5
    
 Pruning Criterion : L2 - norm 
 
| Pruning   ratio |  lamda1  | lamda2 | acc(Ours) | acc(NM) | acc(prune) |
|:---------------:|:--------:|:------:|:---------:|:-------:|:----------:|
|       10%       | 0.000006 | 0.0001 |   92.04   |  91.93  |    89.43   |
|       20%       | 0.000004 |  0.006 |   87.84   |  87.24  |    71.77   |
|       30%       | 0.000001 |  0.01  |   83.25   |  76.91  |    56.95   |
|       40%       | 0.000002 |  0.01  |   66.81   |  54.32  |    31.74   |
|       50%       | 0.00004  | 0.0002 |   45.71   |  32.58  |    12.37   |

### ResNet50 on CIFAR-100
We only provide implementation of Coreset in ResNet-50 on CIFAR-100 because authors of Coreset did not offer the implementation on CNNs. If you test the Coreset, run the below command 

    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type coreset --dataset cifar100 --pruning-ratio 0.1


The following results can be reproduced with command:

    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type OURS --dataset cifar100 --criterion l2-norm --lamda-1 0.00002 --lamda-2 0.006 --pruning-ratio 0.1
    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type OURS --dataset cifar100 --criterion l2-norm --lamda-1 0.00001 --lamda-2 0.002 --pruning-ratio 0.2
    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type OURS --dataset cifar100 --criterion l2-norm --lamda-1 0.00001 --lamda-2 0.002 --pruning-ratio 0.3
    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type OURS --dataset cifar100 --criterion l2-norm --lamda-1 0.00001 --lamda-2 0.001 --pruning-ratio 0.4
    python main.py --arch ResNet50 --pretrained ./saved_models/ResNet.cifar100.original.50.pth.tar --model-type OURS --dataset cifar100 --criterion l2-norm --lamda-1 0.000001 --lamda-2 0.001 --pruning-ratio 0.5
    
 Pruning Criterion : L2 - norm 

| Pruning ratio | lamda1 | lamda2 | acc(Ours) | acc(NM) | acc(prune) |
|:-------------:|:------:|:------:|:---------:|:-------:|:----------:|
|       10%       |  0.00002 |  0.006 |   78.14   |  77.28  |    75.14   |
|       20%       |  0.00001 |  0.002 |   76.15   |  72.73  |    63.39   |
|       30%       |  0.00001 |  0.002 |   73.29   |  64.47  |    39.96   |
|       40%       |  0.00001 |  0.001 |   65.21   |   46.4  |    15.32   |
|       50%       | 0.000001 |  0.001 |   52.61   |  25.98  |    5.22    |

### ResNet34 on ImageNet
The following results can be reproduced with command:

    python main.py --arch ResNet34 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.00007 --lamda-2 0.05 --pruning-ratio 0.1
    python main.py --arch ResNet34 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.00002 --lamda-2 0.07 --pruning-ratio 0.2
    python main.py --arch ResNet34 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.0005 --lamda-2 0.03 --pruning-ratio 0.3
   
Pruning Criterion : L2 - norm 

| Pruning   ratio |  lamda1  | lamda2 | acc(Ours) |   acc(NM)  | acc(prune) |
|:---------------:|:--------:|:------:|:---------:|:----------:|:----------:|
|       10%       |  0.00007 |  0.05  |   69.22   |      66.96 |    63.74   |
|       20%       |  0.00002 |  0.07  |   62.49   |       55.7 |    42.81   |
|       30%       |  0.0005  |  0.03  |   47.59   |      39.22 |    17.02   |


### ResNet101 on ImageNet
The following results can be reproduced with command:

    python main.py --arch ResNet101 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.000004 --lamda-2 0.02 --pruning-ratio 0.1
    python main.py --arch ResNet101 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.000001 --lamda-2 0.02 --pruning-ratio 0.2
    python main.py --arch ResNet101 --model-type OURS --dataset ImageNet --criterion l2-norm --lamda-1 0.000002 --lamda-2 0.03 --pruning-ratio 0.3

Pruning Criterion : L2 - norm 

| Pruning   ratio |  lamda1  | lamda2 | acc(Ours) | acc(NM) | acc(prune) |
|:---------------:|:--------:|:------:|:---------:|:-------:|:----------:|
|       10%       | 0.000004 |  0.02  |   74.59   |  72.36  |    68.9    |
|       20%       | 0.000001 |  0.02  |   68.47   |  61.42  |    45.78   |
|       30%       | 0.000002 |  0.03  |   55.51   |  37.38  |    10.32   |


### Hyperparameters 

<p align="center">
<img src="https://github.com/ICML2022-LBYL/LBYL2022/blob/main/images/LBYL_hyperparams.png" width="60%" height="60%">
</p>


## 3 Our experimental setting
We use NVIDIA Quadro  RTX  6000  GPU  and  Intel  Core  Xeon  Gold5122

