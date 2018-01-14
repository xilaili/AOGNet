#!/usr/bin/env python

import os
import numpy as np
import re
import argparse

res = re.compile('.*Epoch\[(\d+)\] Validation-accuracy.*=([.\d]+)')
param = re.compile('total number of params: ([.\d]+) M')


def get_acc(log_name, top=1):

    name = log_name.replace(".log", "")

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    check_param = True
    param_size = 0
    for l in lines:
        if check_param and param.match(l) is not None:
            param_size = float(param.match(l).groups()[0])
            check_param = False
        m = res.match(l)
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])
        if epoch not in data:
            data[epoch] = val

    val_err = []
    for k, v in data.items():
        val_err.append([(1.0 - v)*100, k])

    return name, sorted(val_err)[:top], param_size

def main():
    logs = []
    rule = ".*"
    for k in args.keyword.split(','):
        rule += (k + ".*")
    pattern = re.compile(rule + ".log")
    for file in os.listdir(args.dir):
        if pattern.match(file):
            err = get_acc(os.path.join(args.dir, file), args.top)
            print err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get top n validation results from certain run')
    parser.add_argument('--dir', type=str, default='./', help='log dir')
    parser.add_argument('--keyword', type=str, default="", help='keyword')
    parser.add_argument('--top', type=int, default=1, help='top n acc')
    args = parser.parse_args()
    main()
