AOGNet using MXNet
=====================================
## Install MXNet
```shell
./code_sync_compile.sh
```
## How to Train

### imagenet
```shell
python train_aognet.py --cfg cfgs/aog_imagenet.yaml
```

### cifar10
```shell
python train_aognet.py --cfg cfgs/aog_cifar10.yaml
```

### cifar100
```shell
python train_aognet.py --cfg cfgs/aog_cifar100.yaml
```
## Train Results
Current best testing error on cifar10 and cifar100

| dataset       | cifar10 | cifar10+ | cifar100 | cifar100+ |
| :------------ | :-----: | :------: | :------: | :-------: |
| error (%)     |         | 96.15    |          |   81.65   |

'+' means standard data augmentation with only random crop and mirroring.
