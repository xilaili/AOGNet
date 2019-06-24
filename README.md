## The new version code and pretrained models are available at [https://github.com/iVMCL/AOGNets] (https://github.com/iVMCL/AOGNets), and more code and models will be released. 
# Learning Deep Compositional Grammatical Architectures for Visual Recognition
This repository contains the code (in MXNet) for our CVPR2019 paper: "[Learning Deep Compositional Grammatical Architectures for Visual Recognition](https://arxiv.org/abs/1711.05847)" paper by [Xilai Li](https://xilaili.github.io), [Tianfu Wu](http://www4.ncsu.edu/~twu19/)\*, Xi Song. (* Corresponding Author)

### Citation

If you find our project useful in your research, please consider citing:

```
@article{li2017aognet,
  title={Learning Deep Compositional Grammatical Architectures for Visual Recognition},
  author={Xilai Li, Tianfu Wu, Xi Song, Hamid Krim},
  journal={arXiv preprint arXiv:1711.05847},
  year={2017}
}
```

## Contents

1. [Introduction](#introduction)
4. [Contacts](#contacts)

## Introduction
An AOGNet consists of a number of stages each of which is composed of a number of AOG building blocks. An AOG building block is designed based on a principled AND-OR grammar and represented by a hierarchical and compositional AND-OR graph. There are three types of nodes: an AND-node explores composition, whose input is computed by concatenating features of its child nodes; an OR-node represents alternative ways of composition in the spirit of exploitation, whose input is the element-wise sum of features of its child nodes; and a Terminal-node takes as input a channel-wise slice of the input feature map of the AOG building block. AOGNets aim to harness the best of two worlds (grammar models and deep neural networks) in representation learning with end-to-end training.
<img src="https://raw.githubusercontent.com/xilaili/xilaili.github.io/master/images/AOGNet-BuildingBlock.png">



## Contacts
email: xli47@ncsu.edu

Any discussions and contribution are welcomed!
