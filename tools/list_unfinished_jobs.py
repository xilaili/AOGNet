#!/usr/bin/env python

import os
import numpy as np
import re
import argparse

res = re.compile('.*Epoch\[(\d+)\] Validation-accuracy.*=([.\d]+)')
total_epoch = re.compile('.*num_epoch.*: (\d+),.*')
gpu = re.compile(".*'gpus'.*")
time_cost = re.compile('.*Time cost=([.\d]+)')


def check(log_name):
    with open(log_name) as f:
        lines = f.readlines()
    check_tot_epoch = True
    check_gpu = True

    epoch = 0
    time_per_epoch = None
    for l in lines:
        if check_gpu and gpu.match(l) is not None:
            gpu_res = ''.join(l.split(',')[:-1]).replace("'", "")
            check_gpu = False
            continue
        if check_tot_epoch and total_epoch.match(l) is not None:
            tot_epoch = int(total_epoch.match(l).groups()[0])
            check_tot_epoch = False
            continue
        m = res.match(l)
        if m is not None:
            assert len(m.groups()) == 2
            epoch = int(m.groups()[0]) + 1
        t = time_cost.match(l)
        if t is not None:
            assert len(t.groups()) == 1
            time_per_epoch = float(t.groups()[0])
    if epoch == tot_epoch:
        return True
    else:
        if time_per_epoch is not None:
            estimated_time_in_s = int(time_per_epoch * (tot_epoch - epoch))
            h = estimated_time_in_s/3600
            m = (estimated_time_in_s % 3600) / 60
        else:
            h, m = "N/A", "N/A"
        return False, log_name, epoch, tot_epoch, gpu_res, h, m



def main():
    for root, _, files in os.walk(args.dir):
        for f in files:
            if not f.endswith(args.suffix):
                continue
            check_result = check(os.path.join(root, f))
            if check_result != True:
                print(check_result[1] + ":  {} of {} on{}, estimated time remaining {} hours {} minutes.".format(check_result[2], check_result[3], check_result[4], check_result[5], check_result[6]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get top n validation results from certain run')
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--suffix", type=str, default=".log", help="suffix")
    args = parser.parse_args()
    main()
