import numpy as np
from easydict import EasyDict as edict
import yaml

cfg = edict()
cfg.batch_size = 64
cfg.num_epoch = 300

# network
cfg.network = edict()
cfg.network.num_stages = 3
cfg.network.units = [1, 1, 1]                # length should be n_stages
cfg.network.filter_list = [16, 64, 128, 256] # length should be (n_stages + 1)
cfg.network.dropout = 0.0

# AOG configuration
cfg.AOG = edict()
cfg.AOG.dims = [4, 4, 4]                 # lenght must be n_stages
cfg.AOG.or_node_op = 'sum'               # choice: ['sum', 'max']
cfg.AOG.TURN_OFF_UNIT_OR_NODE = False    # turn off or-node with size 1
cfg.AOG.Tnode_basic_unit = ''
cfg.AOG.Anode_basic_unit = ''
cfg.AOG.Onode_basic_unit = ''
'''
choices: ['Conv_BN_ReLu', 'Bottleneck_ResNet', 'Bottleneck_ResNeXt']
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
