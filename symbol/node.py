from aognet.aog.aog_1d import *
from .base import *


def Tnode_Op(data, cfg, aog, node, NodeIdtoSym, in_slices, out_slices, stride=(1, 1), bn_mom=0.9, workspace=512, name=''):
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



def Anode_Op(cfg, aog, node, NodeIdtoSym, in_slices, out_slices, bn_mom=0.9, workspace=512, name=''):
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


def Onode_Op(cfg, aog, node, NodeIdtoSym, in_slices, out_slices, bn_mom=0.9, workspace=512, name=''):
    if len(node.child_ids) == 1:
        Onode_out = NodeIdtoSym[node.child_ids[0]]
    else:
        childfeats = []
        for idx in node.child_ids:
            childfeats.append(NodeIdtoSym[idx])

        # merge all the children nodes by sum/avg/max
        if cfg.AOG.or_node_op == 'sum':
            Onode_out = mx.symbol.ElementWiseSum(*childfeats, name=name + 'ONode_{}_sum'.format(node.id))
        elif cfg.AOG.or_node_op == 'avg':
            Onode_out = mx.symbol.ElementWiseSum(*childfeats, name=name + 'ONode_{}_sum'.format(node.id))
            Onode_out = Onode_out / (len(node.child_ids))

    # operation
    arr = aog.primitive_set[node.array_idx]
    num_filters = out_slices[arr.x2 + 1] - out_slices[arr.x1]
    Onode_out = eval(cfg.AOG.Onode_basic_unit)(data=Onode_out, cfg=cfg, num_filters=num_filters,
                                               in_channels=num_filters,
                                               bn_mom=bn_mom, workspace=workspace,
                                               name=name + "Onode_{}".format(node.id))
    NodeIdtoSym[node.id] = Onode_out
