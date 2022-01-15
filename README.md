# LBYL - code
This is the authors' implementation of the following paper: LeaveBeforeYouLeave: Data-Free Restoration of Pruned Neural Networks Without Fine-Tuning

<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave-/blob/main/images/LBYL_figure_1.png" width="100%" height="100%">






# Contents
1. [Requirements](#1-Requirements)<br>
2. [Pre-trained models and Dataset](#2-Pre-trained-models-and-Dataset)<br>
3. [Modified Results](#3-Modified-Results)<br>
4. [Our experimental setting(GPU and CPU)](#4-Our-experimental-setting)<br>
5. [Comparison on absolute scale coefficients between LBYL and NM](#5-Comparison-on-absolute-scale-coefficients-between-LBYL-and-NM)
6. [Fine tuned or trained accuracies of ResNet50 on CIFAR100 using L2norm criterion](#6-Fine-tuned-or-trained-accuracies-of-ResNet50-on-CIFAR100-using-L2norm-criterion)


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


## 3 Modified Results
This paper propose the problem of restoring a pruned CNN in a way free of training data and fine-tuning. In case of random experiments, we implement 3 times with arbitrary pruning criterions. First, we define the fixed indexes in each layers and choose the filters by the indexes. Second, we select the first k filters. Third, we select the last k filters. By the following tables, we *re-present* here all the experimental results appearing in the manuscript, where LBYL **even more clearly outperforms** NM and Pune with large margins.

**[MobileNet-v2]**
Also, we have added new results of a series of experiments using MobileNet-v2, where we prune only the first layer of each block. This scheme can be seen as a naive adaptation of our method for MobileNet-v2, but LBYL still manages to beat NM in most cases.


<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/LBYL_Results.png" width="80%">
</p>

<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/LBYL_RE_BE_WARE.png" width="80%">
</p>


### Arguments
*Required*:
* `--dataset`: Choose datset. *Option*: `cifar10` or `cifar100` or `ImageNet` 
* `--arch` : Choose architecture *Option*: `VGG16` on cifar10 or `ResNet50` on cifar100 or `ResNet34` on ImageNet or `ResNet101` on ImageNet
* `--model-type`: Choose model type *Option*: `OURS` or `merge` or `prune`  
* `--criterion` : Choose criterion *Option*: `l2-norm` or `l2-GM` or `l1-norm` or `random_1` or `random_2` or `random_3`
* `--lamda-1` : Choose lambda_1 
* `--lamda-2` : Choose lambda_2
* `--pruning-ratio` : Choose pruning ratio 

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
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/LBYL_hyperparams.png" width="60%" height="60%">
</p>


## 4 Our experimental setting
We use NVIDIA Quadro  RTX  6000  GPU  and  Intel  Core  Xeon  Gold5122


## 5 Comparison on absolute scale coefficients between LBYL and NM
Even though it is not trivial to quantify how much information in the preserved filter is lost as a result of the restoration process, we can claim that such a side effect is not much in LBYL, compared to NM. This is because LBYL minimizes the amount of those changes in remaining filters by making as many filters as possible to participate in the restoration process. On the other hand, NM forces each pruned filter to deliver its information to only one remaining filter, which can dramatically change the role of the remaining filter. As shown in the table below, the average and maximum of absolute scaling factors (i.e., coefficients) of LBYL are much smaller than those of NM. More specifically, each un-pruned filter in LBYL is multiplied by only 0.00005 on the average, and consequently the information loss of the filter cannot be significant.

<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/LBYL_NM_Scale_Comparison.png" width="60%" height="60%">
</p>

## Final accuracies of fine-tuned or trained models with ResNet50 on CIFAR100 using L2-norm criterion
The following table shows the experimental results on the final accuracy of each pruned, restored or trained model after fine-tuning or training from scratch. We fine-tune and train each model for 20 epochs, and also show the resulting accuracy of the model trained from scratch after 80 epochs for the reference.

<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/Fine-tuned%20or%20trained%20accuracies%20of%20ResNet-50%20on%20CIFAR100%20-%202.png" width="70%" height="60%">
</p>

<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/Fine-tuned%20or%20trained%20accuracies%20of%20ResNet-50%20on%20CIFAR100%20-%201.png" width="70%" height="60%">
</p>

LBYL shows the fastest convergence speed during the fine-tuning or training process as shown in the following training curves.

<p align="center">
<img src="https://github.com/LBYL-2021/LeaveBeforeYouLeave/blob/main/images/learning%20curve.png" width="100%" height="70%">
</p>
