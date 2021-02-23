#!/usr/bin/env python3

# TODO: Write script to run inference using resultant model

import sys
#import torch
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import graphviz;
import numpy as np;
import pickle
from numpy import genfromtxt;
from sklearn import linear_model;
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold;
from sklearn.model_selection import train_test_split;
from sklearn.model_selection import cross_val_score;
from sklearn import ensemble;
from sklearn import tree;
from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import math
import argparse
import sys
from distutils.util import strtobool
from tabulate import tabulate
#################### CONFIG ####################
filtered_names = ["Cycle", "Global_Distance", "Workload"]
result_names = ["Power_mW"]
hidden_layer_sizes = [10, 10, 10, 10, 10]
use_cuda = False
normalize = True
zero_center = True
batch_size = 20000
################################################

"""
class Fcnn(torch.nn.Module):
	def __init__(self, inputs, outputs, hidden_sizes):
		super(Fcnn, self).__init__()
		self.layer_sizes = [inputs] + hidden_sizes + [outputs]
		self.layers = torch.nn.ModuleList()
		for i in range(len(self.layer_sizes)-1):
			self.layers.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
			#self.layers[-1].weight = torch.nn.Parameter(torch.randn(self.layer_sizes[i], self.layer_sizes[i+1]))
			#self.layers[-1].bias = torch.nn.Parameter(torch.randn(self.layer_sizes[i+1]))
			torch.nn.init.normal_(self.layers[-1].weight)
			torch.nn.init.normal_(self.layers[-1].bias)
	def forward(self, in_data):
		intermediate = in_data.clone()
		for layer in self.layers:
			intermediate = torch.nn.functional.relu(layer(intermediate)) # run layer forward, ReLU activation
		return intermediate
"""		
def two_mode_m_predict(power_test):
  power_threshold = np.amin(power_test)*1.1;
  m = [];
  iter_cnt = 0;
  for i in np.nditer(power_test):
      if(i>power_threshold):
          m.append(power_test[iter_cnt]);
      iter_cnt = iter_cnt+1;
    
  power_mean = np.mean(np.asarray(m));
  l = [];
  power_min=np.amin(power_test);
  for i in np.nditer(power_test):
    if(i>power_threshold):
      l.append(power_mean);
    else:
      l.append(power_min);
  return(np.asarray(l));


def return_mape_2(p_buf_power_test,p_buf_pred):
    norm_abs_diff = abs(p_buf_power_test - p_buf_pred)/p_buf_power_test;
    mape_2 =  np.mean((norm_abs_diff))*100;
    return mape_2

def return_mape(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    mean_abs_diff = sum(abs_diff)/len(abs_diff);
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape

def return_maxerror(p_buf_power_test,p_buf_pred):
    max_error = (np.amax(abs(p_buf_power_test-p_buf_pred)*100/np.mean(p_buf_power_test)));
    index = (np.argmax(abs(p_buf_power_test-p_buf_pred)*100/np.mean(p_buf_power_test)));
    test_pwr_with_max_error = p_buf_power_test[index];
    pred_pwr_with_max_error = p_buf_pred[index];
    return [max_error, test_pwr_with_max_error, pred_pwr_with_max_error]

def return_avgerror(p_buf_power_test,p_buf_pred):
    abs_diff = (p_buf_power_test - p_buf_pred);
    mean_abs_diff = sum(abs_diff)/len(abs_diff);
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape

def return_numzero(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    return (len(abs_diff) - np.count_nonzero(abs_diff));

def return_mape_nz(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    mean_abs_diff = sum(abs_diff)/(np.count_nonzero(abs_diff));
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape


parser = argparse.ArgumentParser(description="Run simple regression models and return MAPE in csv format {LR,LR-L,LR-R,LR-B,LR-EN,DT,GB,RF}")
parser.add_argument("-te","--test_csv", required=True, help="Test CSV")
parser.add_argument("-fl","--features_list", required=True, help="Features_list number")
parser.add_argument("-o","--output", required=True,help="Output file with accuracy numbers printed")
parser.add_argument("-m","--model", required=True,help="Model to invoc")

args = parser.parse_args()
output_text = args.output + ".rpt";
sys.stdout = open(output_text, "w+");

# Load from file
print("Opening model :"+str(args.model));
#with open(args.model, "rb") as f:
#    rawdata = f.read()
    
dt_m = joblib.load(args.model)


#Read samples
test_samples = pd.read_csv(args.test_csv);
features = [x.strip() for x in args.features_list.split(',')];
features_print = ['Cycle','Benchmark'];
features_test = test_samples[features];
power_test = test_samples['Power_mW'];
cycle_test = test_samples[features_print];

#Find accuracy
dt_pred = dt_m.predict(features_test);
lr_pred_df = pd.DataFrame(dt_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".dt_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_dt_m = return_mape(power_test,dt_pred);
maep_nz_dt_m = return_mape_nz(power_test,dt_pred);
mape_dt_m = return_mape_2(power_test,dt_pred);
maxerror_dt_m = return_maxerror(power_test,dt_pred);
avgerror_dt_m = return_avgerror(power_test,dt_pred);
numzero_dt_m = return_numzero(power_test,dt_pred);



print "----------------------------------------------------------------------------------------";
print "Results";
print "----------------------------------------------------------------------------------------";
table = [["MAEP" , str(round(maep_dt_m,2))]];
table.append(["Max_error based on MAEP" , str(round(maxerror_dt_m[0],2))]);
table.append(["Avg_error based on MAEP" , str(round(avgerror_dt_m,2)) ]);
print "\n---------------------------------------------------------------------------------------------------------------------------------------------------";
print(tabulate(table,headers=["Model" , "DT"], tablefmt='orgtbl'));
print "---------------------------------------------------------------------------------------------------------------------------------------------------\n\n";
