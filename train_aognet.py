import argparse,logging,os,pprint
import mxnet as mx
import symbol.symbol_aognet as AOGNet
import aognet.utils.memonger
from aognet.aog.aog_1d import get_aog
from aognet.cfgs.config import cfg, read_cfg
from aognet.loader import *
from aognet.utils.scheduler import multi_factor_scheduler

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    # read config
    read_cfg(args.cfg)
    if args.gpus:
        cfg.gpus = args.gpus
    if args.model_path:
        cfg.model_path = args.model_path
    pprint.pprint(cfg)

    # get aogs
    aogs = []
    for i in range(len(cfg.AOG.dims)):
        aog = get_aog(dim=cfg.AOG.dims[i], min_size=cfg.AOG.min_sizes[i], tnode_max_size=cfg.AOG.tnode_max_size[i],
                      turn_off_unit_or_node=cfg.AOG.TURN_OFF_UNIT_OR_NODE)
        aogs.append(aog)

    # get symbol
    symbol = AOGNet.get_symbol(aogs=aogs, cfg=cfg)
    # check shapes
    internals = symbol.get_internals()
    dshape = (cfg.batch_size, 3, 224, 224) if cfg.dataset.data_type == 'imagenet' else (cfg.batch_size, 3, 32, 32) # (32, 32) for cifar10 and cifar100
    _, out_shapes, _ = internals.infer_shape(data=dshape)
    shape_dict = dict(zip(internals.list_outputs(), out_shapes))
    # pprint.pprint(shape_dict)
    # plt = mx.viz.plot_network(symbol, title="aognet", shape={"data": (1, 1, 32, 32)}, hide_weights=False)
    # plt.render('aognet')

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
            #print k, shape_dict[k], size
            sum += size
    print('total number of params: {} M'.format(sum / 1e6))
    for k, v in stages_kw.items():
        print('{} has param size: {} M'.format(k, v / 1e6))

    # setup memonger
    if cfg.memonger:
        dshape_ = (1,) + dshape[1:]
        if args.no_run:
            old_cost = memonger.get_cost(symbol, data=dshape_)
        symbol = memonger.search_plan(symbol, data=dshape_)
        if args.no_run:
            new_cost = memonger.get_cost(symbol, data=dshape_)
            print('batch size=1, old cost= {} MB, new cost= {} MB'.format(old_cost, new_cost))

    if args.no_run:
        return


    # training setup
    kv = mx.kvstore.create(cfg.kv_store)
    devs = mx.cpu() if cfg.gpus is None else [mx.gpu(int(i)) for i in cfg.gpus.split(',')]
    epoch_size = max(int(cfg.dataset.num_examples / cfg.batch_size / kv.num_workers), 1)
    begin_epoch = cfg.model_load_epoch if cfg.model_load_epoch else 0
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    model_prefix = cfg.model_path + "/aognet-{}-{}-".format(cfg.dataset.data_type, kv.rank) + cfg.gpus
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if cfg.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, cfg.model_load_epoch)

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
        batch_end_callback = mx.callback.Speedometer(cfg.batch_size, cfg.frequent),
        epoch_end_callback = checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training aognet")
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpus', help='the gpus will be used', type=str, default='')
    parser.add_argument('--model_path', help='the loc to save model checkpoints', default='', type=str)
    parser.add_argument('--no_run', action='store_true', default=False, help='stop before training')
    args = parser.parse_args()
    logging.info(args)
    main()
