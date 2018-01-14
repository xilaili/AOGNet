#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

res = [re.compile('.*Epoch\[(\d+)\] .*Train-accuracy=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Validation-accuracy.*=([.\d]+)')]


def plot_acc(log_name, color="r"):
    if args.cifar10:
        vmin, vmax, resolution, epochs_tot = 0.0, 0.3, 0.01, 300
    elif args.cifar100:
        vmin, vmax, resolution, epochs_tot = 0.0, 0.7, 0.05, 300
    elif args.imagenet:
        vmin, vmax, resolution, epochs_tot = 0.0, 0.7, 0.05, 120
    else:
        vmin, vmax, resolution, epochs_tot = 0.0, 1.0, 0.05, 120

    train_name = log_name.replace(".log", " train")
    val_name = log_name.replace(".log", " val")

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:  # i=0, match train acc
                break
            i += 1  # i=1, match validation acc
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])
        if epoch not in data:
            data[epoch] = [0] * len(res) * 2
        data[epoch][i*2] += val  # data[epoch], val:number
        data[epoch][i*2+1] += 1

    train_acc = []
    val_acc = []
    for k, v in data.items():
        if v[1]:
            train_acc.append(1.0 - v[0]/(v[1]))
        if v[2]:
            val_acc.append(1.0 - v[2]/(v[3]))

    x_train = np.arange(len(train_acc))
    x_val = np.arange(len(val_acc))
    plt.plot(x_train, train_acc, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.plot(x_val, val_acc, '-', linestyle='-', color=color, linewidth=2, label=val_name)
    plt.legend(loc="best")
    plt.xticks(np.arange(0, epochs_tot, 10))
    plt.yticks(np.arange(vmin, vmax, resolution))
    plt.xlim([0, epochs_tot])
    plt.ylim([vmin, vmax])

def main():
    plt.figure(figsize=(12, 9))
    plt.xlabel("epoch")
    plt.ylabel("error")
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gold', 'brown']
    if args.logs:
        log_files = [i for i in args.logs.split(',')]
    else:
        log_files = []
        rule = ".*"
        for k in args.keyword.split(','):
            rule += (k + ".*")
        pattern = re.compile(rule + ".log")
        for file in os.listdir(args.dir):
            if pattern.match(file):
                log_files.append(os.path.join(args.dir, file))
                print file


    for c in range(len(log_files)):
        plot_acc(log_files[c], color[c%len(color)])
    plt.grid(True)
    if args.cifar10:
        figname = os.path.join(args.dir, 'cifar10.png')
    elif args.cifar100:
        figname = os.path.join(args.dir, 'cifar100.png')
    elif args.imagenet:
        figname = os.path.join(args.dir, 'imagenet.png')
    plt.savefig(figname)
    os.system("eog " + figname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves, using like: \n'
                                     'python -u plot_curve.py --log=resnet-18.log,resnet-50.log')
    parser.add_argument('--logs', type=str, default='',
                        help='the path of log file, --logs=resnet-50.log,resnet-101.log')
    parser.add_argument('--dir', type=str, default="./", help='log directory')
    parser.add_argument('--keyword', type=str, default="", help='keyword')
    parser.add_argument('--cifar10', action='store_true', default=False)
    parser.add_argument('--cifar100', action='store_true', default=False)
    parser.add_argument('--imagenet', action='store_true', default=False)
    args = parser.parse_args()
    main()
