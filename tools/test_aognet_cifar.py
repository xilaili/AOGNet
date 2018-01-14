#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mxnet as mx
import argparse, logging
from cfgs.config import cfg, read_cfg
from aog.aog_1d import get_aog
import symbol.symbol_aognet as aognet

def main():
    devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    if args.cfg:
        read_cfg(args.cfg)
        # get aogs
        aogs = []
        for i in range(len(cfg.AOG.dims)):
            aog = get_aog(dim=cfg.AOG.dims[i], min_size=cfg.AOG.min_sizes[i], tnode_max_size=cfg.AOG.tnode_max_size[i],
                          turn_off_unit_or_node=cfg.AOG.TURN_OFF_UNIT_OR_NODE)
            aogs.append(aog)

        # get symbol
        symbol = aognet.get_symbol(aogs=aogs, cfg=cfg)
    print("symbol loaded")

    if args.dataset == 'cifar10':
        path_imgrec = "../data/cifar10/cifar10_val.rec"
    elif args.dataset == 'cifar100':
        path_imgrec = "../data/cifar100/cifar100_test.rec"

    label_name = 'softmax_label'

    validation_data_iter = mx.io.ImageRecordIter(
                path_imgrec         = path_imgrec,
                label_width         = 1,
                data_name           = 'data',
                label_name          = label_name,
                batch_size          = 128,
                data_shape          = (3,32,32),
                rand_crop           = False,
                rand_mirror         = False,
                num_parts           = 1,
                part_index          = 0)

    cifar_model = mx.mod.Module(symbol=symbol, context=devs, label_names=[label_name,])
    cifar_model.bind(for_training=False, data_shapes=validation_data_iter.provide_data, label_shapes=validation_data_iter.provide_label)
    cifar_model.set_params(arg_params, aux_params)

    metrics = [mx.metric.create('acc'), mx.metric.create('ce')]
    print("testing!!")
    for batch in validation_data_iter:
        cifar_model.forward(batch, is_train=False)
        for m in metrics:
            cifar_model.update_metric(m, batch.label)

    print("Accuracy: {}, Cross-Entropy: {}".format(metrics[0].get()[1], metrics[1].get()[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training aognet")
    parser.add_argument('--cfg', help='experiment configure file name', default='', type=str)
    parser.add_argument('--gpus', help='the gpus will be used', type=str, default='0')
    parser.add_argument('--prefix', help='model prefix', required=True, type=str)
    parser.add_argument('--epoch', help='load model epoch', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    logging.info(args)
    main()
