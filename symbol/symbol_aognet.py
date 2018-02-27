from .node import Tnode_Op, Onode_Op, Anode_Op
from .base import *
from aognet.aog.aog_1d import *

eps = 2e-5

def aog_unit(data, cfg, aog, in_channels, out_channels, stride=(1, 1), bn_mom=0.9, workspace=512, name=''):

    def calculate_slices(dim, channels):
        slices = [0] * dim
        for i in range(channels):
            slices[i%dim] += 1
        for d in range(1, dim):
            slices[d] += slices[d-1]
        slices = [0] + slices
        return slices

    dim = aog.dim
    in_slices = calculate_slices(dim, in_channels)
    out_slices = calculate_slices(dim, out_channels)

    NodeIdtoSym = {}
    for node in aog.node_set:
        if node.node_type == NodeType.TerminalNode:
            Tnode_Op(data=data, cfg=cfg, aog=aog, node=node, NodeIdtoSym=NodeIdtoSym, in_slices=in_slices,
                     out_slices=out_slices, stride=stride, bn_mom=bn_mom, workspace=workspace, name=name)
    for id in aog.DFS:
        node = aog.node_set[id]
        if node.node_type == NodeType.AndNode:
            Anode_Op(cfg=cfg, aog=aog, node=node, NodeIdtoSym=NodeIdtoSym, in_slices=in_slices,
                     out_slices=out_slices, bn_mom=bn_mom, workspace=workspace, name=name)
        elif node.node_type == NodeType.OrNode:
            Onode_Op(cfg=cfg, aog=aog, node=node, NodeIdtoSym=NodeIdtoSym, in_slices=in_slices,
                     out_slices=out_slices, bn_mom=bn_mom, workspace=workspace, name=name)
    output_feat = NodeIdtoSym[aog.BFS[0]]

    return output_feat


def aog_block(data, cfg, aog, units, in_channels, out_channels, stage, bn_mom=0.9, workspace=512, name=''):
    stride = (2, 2) if stage > 0 else (1, 1)
    data = aog_unit(data=data, cfg=cfg, aog=aog, in_channels=in_channels, out_channels=out_channels, stride=stride,
                    bn_mom=bn_mom, workspace=workspace, name=name + 'unit_0_')
    for i in range(1, units):
        data = aog_unit(data=data, cfg=cfg, aog=aog, in_channels=out_channels, out_channels=out_channels, bn_mom=bn_mom,
                        workspace=workspace, name=name + 'unit_{}_'.format(i))
    return data


def get_symbol(aogs, cfg):
    '''
    :param units: list of aog sizes
    :param num_stages: num of aog blocks
    :param num_classes: number of classes
    :param data_type: dataset name, e.g. imagenet, cifar, ...
    :return:
    '''
    # symbol paramters
    num_stages = cfg.network.num_stages
    filter_list = cfg.network.filter_list
    units = cfg.network.units
    data_type = cfg.dataset.data_type
    num_classes = cfg.dataset.num_classes
    bn_mom = cfg.train.bn_mom
    workspace = cfg.train.workspace

    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, momentum=bn_mom, name='bn_data')

    if data_type == 'imagenet':
        body = conv_bn_relu(data=data, cfg=cfg, num_filters=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), bn_mom=bn_mom,
                            workspace=workspace, name='first')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')
    elif data_type in ['cifar10', 'cifar100']:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    for i in range(num_stages):
        body = aog_block(data=body, cfg=cfg, aog=aogs[i], units=units[i], in_channels=filter_list[i], out_channels=filter_list[i+1],
                         stage=i, bn_mom=bn_mom, workspace=workspace, name='stage_{}_'.format(i))

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')

    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

