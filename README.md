# AOGNets: Deep AND-OR Grammar Network for Visual Recognition
This repository contains the code (in MXNet) for: "[AOGNets: Deep AND-OR Grammar Networks for Visual Recognition](https://arxiv.org/abs/1711.05847)" paper by [Xilai Li](https://xilaili.github.io), [Tianfu Wu](http://www4.ncsu.edu/~twu19/)\*, Xi Song and Hamid Krim. (* Corresponding Author)

### Citation

If you find our project useful in your research, please consider citing:

```
@article{li2017aognet,
  title={AOGNets: Deep AND-OR Grammar Networks for Visual Recognition},
  author={Xilai Li, Tianfu Wu, Xi Song, Hamid Krim},
  journal={arXiv preprint arXiv:1711.05847},
  year={2017}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction
An AOGNet consists of a number of stages each of which is composed of a number of AOG building blocks. An AOG building block is designed based on a principled AND-OR grammar and represented by a hierarchical and compositional AND-OR graph. There are three types of nodes: an AND-node explores composition, whose input is computed by concatenating features of its child nodes; an OR-node represents alternative ways of composition in the spirit of exploitation, whose input is the element-wise sum of features of its child nodes; and a Terminal-node takes as input a channel-wise slice of the input feature map of the AOG building block. AOGNets aim to harness the best of two worlds (grammar models and deep neural networks) in representation learning with end-to-end training.
<img src="https://raw.githubusercontent.com/xilaili/xilaili.github.io/master/images/AOGNet-BuildingBlock.png">

## Usage

### Install MXNet
please follow the official instruction to [install MXNet](https://mxnet.incubator.apache.org/install/index.html).

### Train on CIFAR-10/100 dataset
As an example, use the following command to train an AOGNet on CIFAR-10 with training setup and network configuration defined in [cfgs/cifar10/aognet_cifar10_ps_4_bottleneck_1M.yaml](cfgs/cifar10/aognet_cifar10_ps_4_bottleneck_1M.yaml), using two GPUs (gpu_id=0,1). 
```shell
python main.py --cfg cfgs/cifar10/aognet_cifar10_ps_4_bottleneck_1M.yaml --gpus 0,1
```

### Train on ImageNet-1K dataset

To prepare the trainin dataset (.rec file) for ImageNet-1k, please follow the [mxnet image classfication repo](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#prepare-datasets) or this [mxnet resnet implementation by Tornadomeet](https://github.com/tornadomeet/ResNet).

Use following command to train an AOGNet on ImageNet-1K with training setup and network configuration defined in [cfgs/imagenet/aognet_imagenet_1k_v1.yaml](cfgs/imagenet/aognet_imagenet_1k_v1.yaml), using four GPUs and memonger. Memonger[1] is an effective way to save GPU memory when the GPU resource is limited. 
```shell
python main.py --cfg cfgs/imagenet/aognet_imagenet_1k_v1.yaml --gpus 0,1,2,3 --memonger
```

## Results

### Results on CIFAR

| Model | Params | CIFAR-10 (%) | CIFAR-100 (%)|
|---|---|---|---|
| AOGNet-4-(1,1,1) | 1.0M | 5.29 | 25.98 |
| AOGNet-4-(1,1,1) | 8.1M | 4.02 | 20.64 |
| AOGNet-4-(1,1,1) | 16.0M | 3.79 | 19.50 |
| AOGNet-BN-4-(1,1,1) | 1.0M | 4.74 | 22.81 |
| AOGNet-BN-4-(1,1,1) | 8.0M | 3.99 | 18.71 |
| AOGNet-BN-4-(1,2,1) | 16.0M | 3.78 | 17.82 |

The training is done with standard random crop and flip data augmentation.

### Results on ImageNet

| Model | Params | Top-1 Err. | Top-5 Err. | MXNet Model |
|---|---|---|---|---|
| AOGNet-BN-4-(1,1,1,1) | 79.5M | 21.49 | 5.76 | [Download](https://drive.google.com/open?id=1BWFchuwne-QsItJX10PDv87yGSu0ruL3) |

### Training logs and pretrained models

Our trained models and training logs are downloadable at [Google Drive](https://drive.google.com/open?id=10DqN-ylDF_fFgvFmewnm1NEoqQCa1UAB)


## References
1. Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin, Training Deep Nets with Sublinear Memory Cost, [arXiv:1604.06174](https://arxiv.org/abs/1604.06174), [https://github.com/dmlc/mxnet-memonger](https://github.com/dmlc/mxnet-memonger)


## Contacts
email: xli47@ncsu.edu

Any discussions and contribution are welcomed!
