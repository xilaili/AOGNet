#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# ResNet
# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 \
#       --batch-size 64 --num-classes 10 --num-examples 50000 --gpus=0

## train resnet-50
#python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 152 \
#       --batch-size 64 --gpus=1,2,3 2>&1 | tee -a log/imagenet/temp_resnet.log


# AOGNet
# train cifar10
#python -u train_aognet.py --cfg cfgs/aog_cifar10.yaml 2>&1 | tee -a log/cifar10/temp_gpu_0.log

# train cifar100
#python -u train_aognet.py --cfg cfgs/aog_cifar100.yaml 2>&1 | tee -a log/cifar100/temp_gpu_0.log

# train imagenet
python -u train_aognet.py --cfg cfgs/aog_imagenet.yaml 2>&1 | tee -a log/imagenet/temp.log

