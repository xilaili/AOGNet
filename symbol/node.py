from aognet.aog.aog_1d import *
from .base import *


def get_slice_indice(channels, dim):
    slices = [0] * dim
    for i in range(channels):
        slices[i%dim] += 1
    for d in range(1, dim):
        slices[d] += slices[d-1]
    slices = [0] + slices
    return slices


def Preprocess1(data, aog, channels, workspace, name):
    dim = 4
    kernels = [(1, 1), (3, 3), (3, 3), (5, 5)]
    pads = [(0, 0), (1, 1), (1, 1), (2, 2)]
    slices = get_slice_indice(channels, dim)
    feats = []
    # slice channels
    for i in range(dim):
        sliced = mx.symbol.slice_axis(data=data, axis=1, begin=slices[i], end=slices[i+1], name=name + 'preprocess_slice_'+str(i))
        num_filter = slices[i+1] - slices[i]
        conv = mx.sym.Convolution(data=sliced, num_filter=num_filter, kernel=kernels[i], stride=(1, 1), pad=pads[i], no_bias=True,
                                  workspace=workspace, name=name + 'preprocess_conv_'+str(i))

        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + 'preprocess_bn_' + str(i))
        relu = mx.sym.Activation(data=bn, act_type='relu', name=name + 'preprocess_relu_' + str(i))
        feats.append(relu)
    data = mx.symbol.Concat(*feats, dim=1, name=name + 'preprocess_concat')
    return data


def Preprocess2(data, aog, channels, workspace, name):
    dim = aog.dim
    kernels = [(1, 1), (3, 3), (3, 3), (5, 5)]
    pads = [(0, 0), (1, 1), (1, 1), (2, 2)]
    slices = get_slice_indice(channels, len(kernels))
    feats = [[] for _ in range(dim)]
    # slice channels
    for i in range(len(kernels)):
        sliced = mx.symbol.slice_axis(data=data, axis=1, begin=slices[i], end=slices[i+1], name=name + 'preprocess_slice_'+str(i))
        num_filter = slices[i+1] - slices[i]
        conv = mx.sym.Convolution(data=sliced, num_filter=num_filter, kernel=kernels[i], stride=(1, 1), pad=pads[i], no_bias=True,
                                  workspace=workspace, name=name + 'preprocess_conv_'+str(i))
        slices_ = get_slice_indice(num_filter, dim)
        for j in range(dim):
            sliced_ = mx.symbol.slice_axis(data=conv, axis=1, begin=slices_[j], end=slices_[j+1], name=name + 'preprocess_slice_'+str(i)+str(j))
            feats[j].append(sliced_)

    for i in range(dim):
        feats[i] = mx.symbol.Concat(*feats[i], dim=1, name=name + 'preprocess_concat_' + str(i))
    data = mx.symbol.Concat(*feats, dim=1, name=name + 'preprocess_concat')
    return data


def se_nodewise(childfeats, num_filter, cfg, name):
    # concatenation
    concat = mx.symbol.Concat(*childfeats, dim=1, name=name + 'se_concat')
    # squeeze
    squeeze = mx.sym.Pooling(data=concat, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + 'se_squeeze')
    squeeze = mx.symbol.Flatten(data=squeeze, name = name + 'se_flatten')
    # excitation
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*len(childfeats)/16.0), name=name + 'se_fc1')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + 'se_relu')
    excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=len(childfeats), name=name + 'se_fc2')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + 'se_sigmoid')

    res = []
    for i, ch in enumerate(childfeats):
        weight = mx.symbol.slice_axis(data=excitation, axis=1, begin=i, end=i+1, name=name + 'weight_' + str(i))
        weighted = mx.symbol.broadcast_mul(ch, mx.symbol.reshape(data=weight, shape=(-1, 1, 1, 1)))
        res.append(weighted)
    return res


def se_node_and_channelwise(childfeats, num_filter, cfg, name):
    # concatenation
    concat = mx.symbol.Concat(*childfeats, dim=1, name=name + 'se_concat')
    # squeeze
    squeeze = mx.sym.Pooling(data=concat, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + 'se_squeeze')
    squeeze = mx.symbol.Flatten(data=squeeze, name = name + 'se_flatten')
    # excitation
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*len(childfeats)/16.0), name=name + 'se_fc1')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + 'se_relu')
    excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter*len(childfeats), name=name + 'se_fc2')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + 'se_sigmoid')

    res = []
    for i, ch in enumerate(childfeats):
        weight = mx.symbol.slice_axis(data=excitation, axis=1, begin=i*num_filter, end=(i+1)*num_filter, name=name + 'weight_' + str(i))
        weighted = mx.symbol.broadcast_mul(ch, mx.symbol.reshape(data=weight, shape=(-1, num_filter, 1, 1)))
        res.append(weighted)
    return res


def random_weight(childfeats, num_filter, cfg, name):
    length = len(childfeats)
    rand_weights = mx.symbol.random.uniform(low=0, high=1, shape=(1, length))
    normalize_factor = mx.symbol.sum(data=rand_weights, axis=1, name=name+'normalize_factor')
    factor_reshape = mx.symbol.reshape(data=normalize_factor, shape=(1, 1))
    normalized_rand_weights = mx.symbol.broadcast_div(rand_weights, factor_reshape, name=name + 'normalized_rand_weights')

    res = []
    for i, ch in enumerate(childfeats):
        weight = mx.symbol.slice_axis(data=normalized_rand_weights, axis=1, begin=i, end=i+1, name=name + 'weight_' + str(i))
        weighted = mx.symbol.broadcast_mul(ch, mx.symbol.reshape(data=weight, shape=(-1, 1, 1, 1)))
        res.append(weighted)
    return res


def Tnode_Op1(data, cfg, aog, node, NodeIdtoSym, in_slices, out_slices, stride=(1, 1), bn_mom=0.9, workspace=512, name=''):
    ''' slice channel first and perform operation on each slice '''
    # slice
    arr = aog.primitive_set[node.array_idx]
    in_channels = in_slices[arr.x2 + 1] - in_slices[arr.x1]
    Tnode_feat = mx.symbol.slice_axis(data=data, axis=1, begin=in_slices[arr.x1], end=in_slices[arr.x2 + 1],
                                      name=name + 'Tnode_{}_feat_slice'.format(node.id))
    # perform an operation
    out_channels = out_slices[arr.x2 + 1] - out_slices[arr.x1]
    Tnode_out = eval(cfg.AOG.Tnode_basic_unit)(data=Tnode_feat, cfg=cfg, num_filters=out_channels, in_channels=in_channels,
                                               stride=stride, bn_mom=bn_mom, workspace=workspace,
                                               name=name + "Tnode_{}".format(node.id))
    NodeIdtoSym[node.id] = Tnode_out


def Tnode_Op2(data, cfg, aog, node, NodeIdtoSym, in_slices, out_slices, stride=(1, 1), bn_mom=0.9, workspace=512, name=''):
    '''
    using 1x1 conv to get slices and perform operation on each slice
    need more parameters than Op1
    '''
    # slice
    arr = aog.primitive_set[node.array_idx]
    in_channels = in_slices[arr.x2 + 1] - in_slices[arr.x1]
    Tnode_feat = conv_bn_relu(data=data, cfg=cfg, num_filters=in_channels, kernel=(1, 1), pad=(0, 0), bn_mom=bn_mom,
                              workspace=workspace, name=name + "Tnode_{}_feat".format(node.id))
    # perform an operation
    out_channels = out_slices[arr.x2 + 1] - out_slices[arr.x1]
    Tnode_out = eval(cfg.AOG.Tnode_basic_unit)(data=Tnode_feat, cfg=cfg, num_filters=out_channels, in_channels=in_channels,
                                               stride=stride, bn_mom=bn_mom, workspace=workspace,
                                               name=name + "Tnode_{}".format(node.id))
    NodeIdtoSym[node.id] = Tnode_out


def Anode_Op1(cfg, aog, node, NodeIdtoSym, in_slices, out_slices, bn_mom=0.9, workspace=512, name=''):
    '''
    concat two children and perform an opertion
    '''
    assert len(node.child_ids) == 2, "And node has exactly two childs"
    # child[0] is left, child[1] is right
    left_feat = NodeIdtoSym[node.child_ids[0]]
    right_feat = NodeIdtoSym[node.child_ids[1]]
    # concatenate
    Anode_concat = mx.symbol.Concat(*[left_feat, right_feat], dim=1, name=name + 'ANode_{}_concat'.format(node.id))
    arr = aog.primitive_set[node.array_idx]
    num_filters = out_slices[arr.x2 + 1] - out_slices[arr.x1]
    # perform an operation
    Anode_out = eval(cfg.AOG.Anode_basic_unit)(data=Anode_concat, cfg=cfg, num_filters=num_filters,
                                               in_channels=num_filters,
                                               bn_mom=bn_mom, workspace=workspace,
                                               name=name + "Anode_{}".format(node.id))
    NodeIdtoSym[node.id] = Anode_out


def Anode_Op2(cfg, aog, node, NodeIdtoSym, in_slices, out_slices, bn_mom=0.9, workspace=512, name=''):
    '''
    do conv for each child first, then concat
    '''
    assert len(node.child_ids) == 2, "And node has exactly two childs"
    # child[0] is left, child[1] is right
    left_feat = NodeIdtoSym[node.child_ids[0]]
    right_feat = NodeIdtoSym[node.child_ids[1]]
    left_node = aog.node_set[node.child_ids[0]]
    right_node = aog.node_set[node.child_ids[1]]
    arr1 = aog.primitive_set[left_node.array_idx]
    num_filters1 = out_slices[arr1.x2 + 1] - out_slices[arr1.x1]
    arr2 = aog.primitive_set[right_node.array_idx]
    num_filters2 = out_slices[arr2.x2 + 1] - out_slices[arr2.x1]
    # operation on both children
    left_feat = eval(cfg.AOG.Anode_basic_unit)(data=left_feat, cfg=cfg, num_filters=num_filters1, in_channels=num_filters1,
                                               bn_mom=bn_mom, workspace=workspace, name=name + "Anode_{}_left".format(node.id))
    right_feat = eval(cfg.AOG.Anode_basic_unit)(data=right_feat, cfg=cfg, num_filters=num_filters2, in_channels=num_filters2,
                                               bn_mom=bn_mom, workspace=workspace, name=name + "Anode_{}_right".format(node.id))
    # concatenate
    Anode_concat = mx.symbol.Concat(*[left_feat, right_feat], dim=1, name=name + 'ANode_{}_concat'.format(node.id))

    NodeIdtoSym[node.id] = Anode_concat


def Onode_Op1(cfg, aog, node, NodeIdtoSym, in_slices, out_slices, bn_mom=0.9, workspace=512, name=''):
    if len(node.child_ids) == 1:
        Onode_out = NodeIdtoSym[node.child_ids[0]]
    else:
        childfeats = []
        for idx in node.child_ids:
            childfeats.append(NodeIdtoSym[idx])

        # process or_node_weight
        if cfg.AOG.or_node_weight_type:
            arr = aog.primitive_set[node.array_idx]
            num_filters = out_slices[arr.x2 + 1] - out_slices[arr.x1]
            childfeats = eval(cfg.AOG.or_node_weight_type)(childfeats, num_filters, cfg, name + 'ONode_{}_'.format(node.id))

        # merge all the children nodes by sum/avg/max
        if cfg.AOG.or_node_op == 'sum':
            Onode_out = mx.symbol.ElementWiseSum(*childfeats, name=name + 'ONode_{}_sum'.format(node.id))
        elif cfg.AOG.or_node_op == 'avg':
            Onode_out = mx.symbol.ElementWiseSum(*childfeats, name=name + 'ONode_{}_sum'.format(node.id))
            Onode_out = Onode_out / (len(node.child_ids))
        elif cfg.AOG.or_node_op == 'avg_max':
            # Onode_out = childfeats[0]
            # for ch in childfeats[1:]:
            #     Onode_out = mx.symbol.maximum(Onode_out, ch)
            Onode_out_sum = mx.symbol.ElementWiseSum(*childfeats, name=name + 'ONode_{}_sum'.format(node.id))
            Onode_out_sum = Onode_out_sum / (len(node.child_ids))
            Onode_out_max, _ = mx.contrib.symbol.RGMMax(*childfeats, num_args=len(node.child_ids), node_idx=node.id,
                                                 name='ONode_{}_max'.format(node.id))
            Onode_out = (Onode_out_sum + Onode_out_max) / 2.0

    # operation
    if cfg.AOG.USE_OR_NODE_CONV:
        arr = aog.primitive_set[node.array_idx]
        num_filters = out_slices[arr.x2 + 1] - out_slices[arr.x1]
        Onode_out = eval(cfg.AOG.Onode_basic_unit)(data=Onode_out, cfg=cfg, num_filters=num_filters,
                                                   in_channels=num_filters,
                                                   bn_mom=bn_mom, workspace=workspace,
                                                   name=name + "Onode_{}".format(node.id))
    NodeIdtoSym[node.id] = Onode_out
