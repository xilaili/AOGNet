import mxnet as mx


def conv_bn(data, cfg, num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), group=1, workspace=512, bn_mom=0.9, name=''):
    body = mx.sym.Convolution(data=data, num_filter=num_filters, kernel=kernel, stride=stride, pad=pad, num_group=group,
                                  no_bias=True, workspace=workspace, name=name + "_conv")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')

    return body


def conv_bn_relu(data, cfg, num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), group=1, workspace=512, bn_mom=0.9, name=''):
    body = mx.sym.Convolution(data=data, num_filter=num_filters, kernel=kernel, stride=stride, pad=pad, num_group=group,
                                  no_bias=True, workspace=workspace, name=name + "_conv")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
    body = mx.sym.Activation(data=body, act_type='relu', name=name + '_relu')
    if cfg.network.dropout > 0:
        body = mx.symbol.Dropout(data=body, p=cfg.network.dropout, name=name + '_dp')

    return body


def bn_relu_conv(data, cfg, num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), group=1, workspace=512, bn_mom=0.9, name=''):
    body = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
    body = mx.sym.Activation(data=body, act_type='relu', name=name + '_relu')
    body = mx.sym.Convolution(data=body, num_filter=num_filters, kernel=kernel, stride=stride, pad=pad, num_group=group,
                              no_bias=True, workspace=workspace, name=name + "_conv")
    if cfg.network.dropout > 0:
        body = mx.symbol.Dropout(data=body, p=cfg.network.dropout, name=name + '_dp')

    return body


#######################


# pure conv
def Conv_BN_ReLu(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=stride, pad=(1, 1), bn_mom=bn_mom,
                        workspace=workspace, name=name)
    return body


# conv + sc + bottleneck
def Bottleneck_ResNet(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    ratio = 0.25
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=int(num_filters*ratio), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                        bn_mom=bn_mom, workspace=workspace, name=name + '_bn_top')
    body = conv_bn_relu(data=body, cfg=cfg, num_filters=int(num_filters*ratio), kernel=(3, 3), stride=stride, pad=(1, 1),
                        group=1, bn_mom=bn_mom, workspace=workspace, name=name)
    body = conv_bn(data=body, cfg=cfg, num_filters=num_filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                   bn_mom=bn_mom, workspace=workspace, name=name + '_bn_bottom')
    if in_channels == num_filters:
        sc = data
    else:
        sc = conv_bn(data=data, cfg=cfg, num_filters=num_filters, kernel=(1, 1), stride=stride, pad=(0, 0),
                     bn_mom=bn_mom, workspace=workspace, name=name + "_sc")
    if cfg.memonger:
        sc._set_attr(mirror_stage='True')

    return mx.sym.Activation(data=body + sc, act_type='relu', name=name + '_sum_relu')


# ResNeXt bottleneck
def Bottleneck_ResNeXt(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    ratio, cardinality = 0.5, 4
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=int(num_filters*ratio), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                        bn_mom=bn_mom, workspace=workspace, name=name + '_bn_top')
    body = conv_bn_relu(data=body, cfg=cfg, num_filters=int(num_filters*ratio), kernel=(3, 3), stride=stride, pad=(1, 1),
                        group=cardinality, bn_mom=bn_mom, workspace=workspace, name=name)
    body = conv_bn(data=body, cfg=cfg, num_filters=num_filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                   bn_mom=bn_mom, workspace=workspace, name=name + '_bn_bottom')
    if in_channels == num_filters:
        sc = data
    else:
        sc = conv_bn(data=data, cfg=cfg, num_filters=num_filters, kernel=(1, 1), stride=stride, pad=(0, 0),
                     bn_mom=bn_mom, workspace=workspace, name=name + "_sc")
    if cfg.memonger:
        sc._set_attr(mirror_stage='True')

    return mx.sym.Activation(data=body + sc, act_type='relu', name=name + '_sum_relu')
