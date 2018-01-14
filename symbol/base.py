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


# pure conv
def basic_unit1(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=stride, pad=(1, 1), bn_mom=bn_mom,
                        workspace=workspace, name=name)
    return body


# conv + sc
def basic_unit2(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=stride, pad=(1, 1),
                   bn_mom=bn_mom, workspace=workspace, name=name + "_conv1")
    body = conv_bn(data=body, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn_mom=bn_mom,
                   workspace=workspace, name=name + "_conv2")
    return body

# conv x 2 + sc
def basic_unit3(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=stride, pad=(1, 1),
                   bn_mom=bn_mom, workspace=workspace, name=name + "_conv1")
    body = conv_bn(data=body, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bn_mom=bn_mom,
                   workspace=workspace, name=name + "_conv2")
    if in_channels == num_filters:
        sc = data
    else:
        sc = conv_bn(data=data, cfg=cfg, num_filters=num_filters, kernel=(1, 1), stride=stride, pad=(0, 0),
                     bn_mom=bn_mom, workspace=workspace, name=name + "_sc")
    if cfg.memonger:
        sc._set_attr(mirror_stage='True')

    return mx.sym.Activation(data=body + sc, act_type='relu', name=name + '_sum_relu')


# conv + sc + bottleneck (0.5)
def basic_unit4(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=int(num_filters*0.5), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                        bn_mom=bn_mom, workspace=workspace, name=name + '_bn_top')
    body = conv_bn_relu(data=body, cfg=cfg, num_filters=int(num_filters*0.5), kernel=(3, 3), stride=stride, pad=(1, 1),
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


# conv + sc + bottleneck (0.25)
def basic_unit5(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=int(num_filters*0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                        bn_mom=bn_mom, workspace=workspace, name=name + '_bn_top')
    body = conv_bn_relu(data=body, cfg=cfg, num_filters=int(num_filters*0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
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


# group conv (num_group=4)
def basic_unit6(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=stride, pad=(1, 1), group=4,
                        bn_mom=bn_mom, workspace=workspace, name=name)
    return body


# two convs (n->n/2->n)
def basic_unit7(data, cfg, num_filters, in_channels, stride=(1, 1), workspace=512, bn_mom=0.9, name=''):
    body = conv_bn_relu(data=data, cfg=cfg, num_filters=int(0.5*num_filters), kernel=(3, 3), stride=stride, pad=(1, 1),
                        bn_mom=bn_mom, workspace=workspace, name=name+'_conv1')
    body = conv_bn_relu(data=body, cfg=cfg, num_filters=num_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                        bn_mom=bn_mom, workspace=workspace, name=name+'_conv2')
    return body
