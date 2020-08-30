#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import math;
train_split = 0.90

parser = argparse.ArgumentParser(description="Create train, test, and validation sets")
parser.add_argument("--signals", action="append", required=True)
parser.add_argument("--power", action="append", required=True)
#parser.add_argument("--folds", required=True)
parser.add_argument("--output", required=True)
#parser.add_argument("--only_benchmarks", type=int, required=True)
parser.add_argument("--only_these_signals", type=int, required=False, default=0)
parser.add_argument("--signal_list", type=str, required=False,  default="default_signal")
args = parser.parse_args()

if len(args.signals) != len(args.power):
    print("ERROR: Different number of signal and power file specifications")
    exit()

signal_files = args.signals
power_files = args.power
#folds = int(args.folds);
full_data = None
signal_names = None
for workload_idx in range(len(signal_files)):
    print("Process signal file " + signal_files[workload_idx])
    signal_data = pd.read_csv(signal_files[workload_idx])
    if(args.only_these_signals):
      signal_list = args.signal_list.split(',');
      signal_data = signal_data.loc[:,signal_list];
    """signal_header = signal_data.columns.values.tolist()
    if workload_idx == 0:
        signal_names = signal_header
    else:
        (signal_names) != len(signal_header);
        print("ERROR: Signal file name length mismatch")
        exit()
        for name in signal_header:
            if name not in signal_names:
                print("ERROR: Missing signal name " + name)
                exit()
    data_idx = 0"""
    #print signal_data.columns
    #power_values = []
    print("Process power file " + power_files[workload_idx])
    power_data = pd.read_csv(power_files[workload_idx]);
    #print power_data.columns
    power_data['Power_mW'] = power_data['power_w']*1000;
    power_data['Cycle'] = power_data['time_ns']/3;
    full_data_i = pd.merge(signal_data, power_data,  how='inner', on='Cycle');
    #print full_data_i.columns;
    #print full_data_i;
    full_data = pd.concat([full_data, full_data_i]);
full_data.to_csv(args.output, index=False)
full_data = full_data.sample(frac=1).reset_index(drop=True)    # shuffle
"""
fold_data_size = int(math.ceil(full_data.shape[0]*(1-train_split)));
print fold_data_size;
for i in np.arange(folds):
    train_after_0 = (fold_data_size*i)-1;
    train_before_0 =0;
    train_after_1 = full_data.shape[0]-1;
    train_before_1 = fold_data_size*i+fold_data_size;
    test_before = fold_data_size*i;
    test_after = fold_data_size*i+fold_data_size-1;
    #print "Test : " + str(test_before) + str(test_after);
    #print "Train 0 : " + str(train_before_0) + str(train_after_0);
    #print "Train 1 : " + str(train_before_1) + str(train_after_1);
    
    
    if(train_after_0==-1):
        train_data_0 = full_data.head(0);
    else:
        train_data_0 = full_data.truncate(before = train_before_0,after=train_after_0);
    if(train_before_1>full_data.shape[0]-1):
        train_data_1 = full_data.head(0);
    else:
        train_data_1 = full_data.truncate(before = train_before_1,after=train_after_1);
        
    train_data =  pd.concat([train_data_0,train_data_1])
    test_data = full_data.truncate(before=test_before,after=test_after);
    
    if(args.only_benchmarks==1):
      train_data.to_csv(args.output + "_" +str( i)  + "_train.only_benchmarks.90_10_split.csv", index=False);
      test_data.to_csv(args.output + "_" + str(i) + "_test.only_benchmarks.90_10_split.csv", index=False);
    else :
      train_data.to_csv(args.output + "_" +str( i) + "_train.all_cycles.90_10_split.csv", index=False);
      test_data.to_csv(args.output + "_" + str(i) + "_test.all_cycles.90_10_split.csv", index=False);


split_row = int(full_data.shape[0]*train_split)
train_data = full_data.truncate(after=split_row-1)
test_data = full_data.truncate(before=split_row)
train_data.to_csv(args.output + "_train.csv", index=False)
test_data.to_csv(args.output + "_test.csv", index=False)
"""
