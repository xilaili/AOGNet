# AOGNets: Deep AND-OR Grammar Network for Visual Recognition
This repository contains the code and trained models of:

Xilai Li, Tianfu Wu*, Xi Song, and Hamid Krim, AOGNets: "Deep AND-OR Grammar Networks for Visual Recognition". ([arXiv:1711.05847](https://arxiv.org/abs/1711.05847))

## Install MXNet
```shell
./code_sync_compile.sh
```
## How to Train

### imagenet
```shell
python train_aognet.py --cfg aognet/cfgs/aog_imagenet.yaml
```

### cifar10
```shell
python train_aognet.py --cfg aognet/cfgs/aog_cifar10.yaml
```

### cifar100
```shell
python train_aognet.py --cfg aognet/cfgs/aog_cifar100.yaml
```
## Train Results
Current best testing error on cifar10 and cifar100

| dataset       | cifar10 | cifar10+ | cifar100 | cifar100+ |
| :------------ | :-----: | :------: | :------: | :-------: |
| error (%)     |         |          |          |           |

'+' means standard data augmentation with only random crop and mirroring.
