import os
import sys
import mxnet as mx


def cifar100_iterator(cfg, kv):

    train_rec = os.path.join(cfg.dataset.data_dir, "cifar100_train.rec")
    val_rec = os.path.join(cfg.dataset.data_dir, "cifar100_test.rec")

    train = mx.io.ImageRecordIter(
            path_imgrec         = train_rec,
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = (3, 32, 32),
            batch_size          = cfg.batch_size,
            pad                 = 4,
            fill_value          = 127,
            rand_crop           = True if cfg.dataset.aug_level > 0 else False,
            max_random_scale    = 1.0,
            min_random_scale    = 1.0,
            max_aspect_ratio    = 0,
            random_h            = 0,
            random_s            = 0,
            random_l            = 0,
            max_rotate_angle    = 0 if cfg.dataset.aug_level <= 2 else 10,
            max_shear_ratio     = 0 if cfg.dataset.aug_level <= 2 else 0.1,
            rand_mirror         = True if cfg.dataset.aug_level > 0 else False,
            shuffle             = True if cfg.dataset.aug_level >= 0 else False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    val = mx.io.ImageRecordIter(
            path_imgrec         = val_rec,
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = cfg.batch_size,
            data_shape          = (3, 32, 32),
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    return train, val
