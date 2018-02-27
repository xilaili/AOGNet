import argparse
import logging
import os
import pprint
import mxnet as mx
import symbol.symbol_aognet as AOGNet
import aognet.utils.memonger as memonger
from aognet.aog.aog_1d import get_aog
from cfgs.config import cfg, read_cfg
from aognet.loader import *
from aognet.utils.scheduler import multi_factor_scheduler


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():

    # read config
    read_cfg(args.cfg)
    cfg.memonger = args.memonger
    pprint.pprint(cfg)

    # get symbol
    aogs = []
    for i in range(len(cfg.AOG.dims)):
        aog = get_aog(dim=cfg.AOG.dims[i], min_size=1, tnode_max_size=cfg.AOG.dims[i],
                      turn_off_unit_or_node=cfg.AOG.TURN_OFF_UNIT_OR_NODE)
        aogs.append(aog)

    symbol = AOGNet.get_symbol(aogs=aogs, cfg=cfg)

    # check shapes
    internals = symbol.get_internals()
    if cfg.dataset.data_type == 'imagenet':
        dshape = (cfg.batch_size, 3, 224, 224)
    elif cfg.dataset.data_type in ['cifar10', 'cifar100']:
        dshape = (cfg.batch_size, 3, 32, 32)
    _, out_shapes, _ = internals.infer_shape(data=dshape)
    shape_dict = dict(zip(internals.list_outputs(), out_shapes))

    # count params size
    stages_kw = {'stage_0': 0.0, 'stage_1': 0.0, 'stage_2': 0.0, 'stage_3': 0.0}
    sum = 0.0
    for k in shape_dict.keys():
        if k.split('_')[-1] in ['weight', 'bias', 'gamma', 'beta']:
            size = 1
            for val in shape_dict[k]:
                size *= val
            for key in stages_kw:
                if key in k:
                    stages_kw[key] += size
            sum += size
    print('total number of params: {} M'.format(sum / 1e6))
    for k, v in stages_kw.items():
        if v > 0:
            print('{} has param size: {} M'.format(k, v / 1e6))

    # setup memonger
    if args.memonger:
        dshape_ = (1,) + dshape[1:]
        old_cost = memonger.get_cost(symbol, data=dshape_)
        symbol = memonger.search_plan(symbol, data=dshape_)
        new_cost = memonger.get_cost(symbol, data=dshape_)
        print('batch size=1, old cost= {} MB, new cost= {} MB'.format(old_cost, new_cost))

    # training setup
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(cfg.dataset.num_examples / cfg.batch_size / kv.num_workers), 1)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    model_prefix = os.path.join(args.modeldir, 'checkpoint')
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.resume:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.resume)
        begin_epoch = args.resume
    else:
        begin_epoch = 0

    # iterator
    train, val = eval(cfg.dataset.data_type + "_iterator")(cfg, kv)

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    lr_scheduler = multi_factor_scheduler(begin_epoch, epoch_size, step=cfg.train.lr_steps, factor=0.1)

    optimizer_params = {
        'learning_rate': cfg.train.lr,
        'momentum': cfg.train.mom,
        'wd': cfg.train.wd,
        'lr_scheduler': lr_scheduler
    }

    model = mx.mod.Module(
        context             = devs,
        symbol              = symbol)

    if cfg.dataset.data_type in ["cifar10", "cifar100"]:
        eval_metric = ['acc', 'ce']
    elif cfg.dataset.data_type == 'imagenet':
        eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)]

    model.fit(
        train,
        begin_epoch        = begin_epoch,
        num_epoch          = cfg.num_epoch,
        eval_data          = val,
        eval_metric        = eval_metric,
        kvstore            = kv,
        optimizer          = 'sgd',  # ['sgd', 'nag']
        optimizer_params   = optimizer_params,
        arg_params         = arg_params,
        aux_params         = aux_params,
        initializer        = initializer,
        allow_missing      = True,
        batch_end_callback = mx.callback.Speedometer(cfg.batch_size, args.frequent),
        epoch_end_callback = checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training aognet")
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', help='the gpus will be used', type=str, default='0')
    parser.add_argument('--modeldir', help='the location to save model checkpoints', default='./model', type=str)
    parser.add_argument('--kv-store', help='kv-store', type=str, default='device')
    parser.add_argument('--memonger', action='store_true', default=False, help='use memonger to save gpu memory')
    parser.add_argument('--resume', help='resume training start from epoch --', type=int)
    parser.add_argument('--frequent', help='how many batches per print', default=100, type=int)
    args = parser.parse_args()
    print(args)
    main()
