#!/usr/bin/env python

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse,logging
import mxnet as mx
import symbol.symbol_aognet as aognet
import pprint
from aog.aog_1d import get_aog
from cfgs.config import cfg, read_cfg


def get_params_size(symbol):
    # check shapes
    internals = symbol.get_internals()
    dshape = (cfg.batch_size, 3, 224, 224) if cfg.dataset.data_type == 'imagenet' else (cfg.batch_size, 3, 32, 32)
    _, out_shapes, _ = internals.infer_shape(data=dshape)
    shape_dict = dict(zip(internals.list_outputs(), out_shapes))

    # count params size
    sum = 0.0
    for k in shape_dict.keys():
        if k.split('_')[-1] in ['weight', 'bias', 'beta', 'gamma']:
            size = 1
            for val in shape_dict[k]:
                size *= val
            #print k, shape_dict[k], size
            sum += size
    return sum / 1e6


def search_factor(aogs, param_size):
    filter_list_ori = cfg.network.filter_list
    symbol = aognet.get_symbol(aogs=aogs, cfg=cfg)
    left, right = 1.0, 1.0
    ps_ori = get_params_size(symbol)
    print "original param size: {} MB".format(ps_ori)
    if ps_ori > param_size:
        while True:
            left /= 2
            if args.fix_first:
                cfg.network.filter_list = [filter_list_ori[0]] + [int(left*f) for f in filter_list_ori[1:]]
            else:
                cfg.network.filter_list = [int(left*f) for f in filter_list_ori]
            symbol = aognet.get_symbol(aogs=aogs, cfg=cfg)
            if get_params_size(symbol) <= param_size:
                break
            else:
                right = left
    else:
        while True:
            right *= 2
            if args.fix_first:
                cfg.network.filter_list = [filter_list_ori[0]] + [int(right*f) for f in filter_list_ori[1:]]
            else:
                cfg.network.filter_list = [int(right*f) for f in filter_list_ori]
            symbol = aognet.get_symbol(aogs=aogs, cfg=cfg)
            if get_params_size(symbol) >= param_size:
                break
            else:
                left = right

    pre_filter_list = filter_list_ori
    while True:
        factor = (left + right) / 2.0
        if args.fix_first:
            filter_list = [filter_list_ori[0]] + [int(factor*f) for f in filter_list_ori[1:]]
        else:
            filter_list = [int(f*factor) for f in filter_list_ori]
        if filter_list == pre_filter_list:
            break
        pre_filter_list = filter_list
        cfg.network.filter_list = filter_list
        symbol = aognet.get_symbol(aogs=aogs, cfg=cfg)
        ps = get_params_size(symbol)
        if ps <= param_size:
            left = factor
        else:
            right = factor

    return factor, filter_list, ps



def main():
    read_cfg(args.cfg)
    # get aogs
    aogs = []
    for i in range(len(cfg.AOG.dims)):
        aog = get_aog(dim=cfg.AOG.dims[i], min_size=cfg.AOG.min_sizes[i], tnode_max_size=cfg.AOG.tnode_max_size[i],
                      turn_off_unit_or_node=cfg.AOG.TURN_OFF_UNIT_OR_NODE)
        aogs.append(aog)

    factor, filter_list_adjusted, ps = search_factor(aogs, args.param_size)
    print("factor: {}, adjusted filter list: {}, param_size: {} M".format(factor, filter_list_adjusted, ps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training aognet")
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--param_size', help='expected param size', type=float, required=True)
    parser.add_argument('--fix_first', help='expected param size', action='store_true', default=False)
    args = parser.parse_args()
    logging.info(args)
    main()
