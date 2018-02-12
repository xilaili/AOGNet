import numpy as np
from easydict import EasyDict as edict
import yaml

cfg = edict()
cfg.batch_size = 64
cfg.gpus = '0'
cfg.frequent = 100
cfg.kv_store = 'device'
cfg.memonger = False
cfg.retrain = False
cfg.model_load_epoch = 0
cfg.num_epoch = 300
cfg.model_path = "./model"

# network
cfg.network = edict()
cfg.network.num_stages = 3
cfg.network.units = [1, 1, 1]                # length should be n_stages
cfg.network.filter_list = [16, 64, 128, 256] # length should be (n_stages + 1), important for training
cfg.network.dropout = 0.0                    # good for training

# AOG configuration
cfg.AOG = edict()
cfg.AOG.dims = [4, 4, 4]                 # length should be n_stages
cfg.AOG.min_sizes = [1, 1, 1]            # length should be n_stages
cfg.AOG.tnode_max_size = [4, 4, 4]       # the maximum size of tnode allowed
cfg.AOG.or_node_op = 'sum'               # choice: ['sum', 'max']
cfg.AOG.USE_OR_NODE_CONV = True          # pretty useful
cfg.AOG.TURN_OFF_UNIT_OR_NODE = False    # turn off or-node with size 1, better to use False
cfg.AOG.Preprocess_Op = ''
cfg.AOG.or_node_weight_type = ''         # choices: ['se_nodewise', 'se_node_and_channelwise']
cfg.AOG.Tnode_Op = 'Tnode_Op1'
cfg.AOG.Anode_Op = 'Anode_Op1'
cfg.AOG.Onode_Op = 'Onode_Op1'
cfg.AOG.Tnode_basic_unit = 'basic_unit1'
cfg.AOG.Anode_basic_unit = 'basic_unit1'
cfg.AOG.Onode_basic_unit = 'basic_unit1'
'''
basic_unit1: conv_bn_relu
basic_unit2: conv_bn_relu + sc
basic_unit3: conv_bn_relu x 2 + sc
basic_unit4: bottleneck + sc (ratio=0.5)
basic_unit5: bottleneck + sc (ratio=0.25)
'''

# Train
cfg.train = edict()
cfg.train.bn_mom = 0.9
cfg.train.lr = 0.1
cfg.train.mom = 0.9
cfg.train.wd = 0.0001
cfg.train.workspace = 512
cfg.train.lr_steps = [150, 225]

# dataset
cfg.dataset = edict()
cfg.dataset.data_type = 'cifar10'
cfg.dataset.num_classes = 10
cfg.dataset.data_dir = './data/cifar10'
cfg.dataset.num_examples = 50000
cfg.dataset.aug_level = 1
'''
'level 1: use only random crop and random mirror'
'level 2: add scale/aspect/hsv augmentation based on level 1'
'level 3: add rotation/shear augmentation based on level 2'
'''


def read_cfg(cfg_file):
    with open(cfg_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in cfg[k]:
                            cfg[k][vk] = vv
                        else:
                            raise ValueError("key {} not exist in config.py".format(vk))
                else:
                    cfg[k] = v
            else:
                raise ValueError("key {} exist in config.py".format(k))
