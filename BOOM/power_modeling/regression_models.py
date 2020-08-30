#!/usr/bin/env python3

# TODO: Write script to run inference using resultant model

import sys
#import torch
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import graphviz;
import numpy as np;
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
parser.add_argument("-tr","--train_csv", required=True, help="Training CSV")
parser.add_argument("-te","--test_csv", required=True, help="Test CSV")
parser.add_argument("-lr","--lr_params",required=False, help="fit_intercept,normalize", default="true,true")
parser.add_argument("-lrl","--lrl_params",required=False, help="alpha,fit_intercept,normalize,positive", default="0.1,true,true,true")
parser.add_argument("-lrr","--lrr_params",required=False, help="alpha,fit_intercept,normalize", default="0.1,true,true")
parser.add_argument("-lrb","--lrb_params",required=False, help="alpha_1,alpha_2,lambda_1,lambda_2,fit_intercept,normalize", default="0.0001,0.0001,0.0001,0.0001,true,true")
parser.add_argument("-lren","--lren_params",required=False, help="alpha,l1_ratio,fit_intercept,normalize,positive", default="0.1,0.5,true,true,true")
parser.add_argument("-dt","--dt_params",required=False, help="max_depth,min_samples_split,min_samples_leaf,max_leaf_nodes", default="20,2,1,None")
parser.add_argument("-dt_ten","--dt_ten_params",required=False, help="max_depth,min_samples_split,min_samples_leaf,max_leaf_nodes", default="10,2,1,None")
parser.add_argument("-gb","--gb_params",required=False, help="learning_rate,n_estimators,max_depth,min_samples_split,min_samples_leaf", default="0.1,5,6,2,1")
parser.add_argument("-rf","--rf_params",required=False, help="n_estimators,max_depth,min_samples_split,min_samples_leaf", default="5,6,2,1")
parser.add_argument("-fl","--features_list", required=True, help="Features_list number")
parser.add_argument("-o","--output", required=True,help="Output file with accuracy numbers printed")
parser.add_argument("-p","--print_duplicate",type=int, required=True,help="Print duplicates")
parser.add_argument("-d","--power_high_dupcnt",type=float, required=False,help="Power with high duplicate count", default=200)

args = parser.parse_args()
output_text = args.output + ".rpt";
sys.stdout = open(output_text, "w+");

#Get params and instantiate regression models
print "----------------------------------------------------------------------------------------";
print "Model Parameters:";
print "----------------------------------------------------------------------------------------";
print "LR_params used:"+ args.lr_params;
params_list  = [x.strip() for x in args.lr_params.split(',')];
params = {'fit_intercept':bool(strtobool(params_list[0])), 'normalize': bool(strtobool(params_list[1]))};
lr_m = linear_model.LinearRegression(**params);

print "LRL_params used:"+ args.lrl_params;
params_list  = [x.strip() for x in args.lrl_params.split(',')];
params = {'alpha':float(params_list[0]),'fit_intercept':bool(strtobool(params_list[1])), 'normalize': bool(strtobool(params_list[2])),'positive':bool(strtobool(params_list[3]))};
lrl_m = linear_model.Lasso(**params);

print "LRR_params used:"+ args.lrr_params;
params_list  = [x.strip() for x in args.lrr_params.split(',')];
params = {'alpha':float(params_list[0]),'fit_intercept':bool(strtobool(params_list[1])), 'normalize': bool(strtobool(params_list[2]))};
lrr_m = linear_model.Ridge(**params);

print "LRB_params used:"+ args.lrb_params;
params_list  = [x.strip() for x in args.lrb_params.split(',')];
params = {'alpha_1':float(params_list[0]),'alpha_2':float(params_list[1]),'lambda_1':float(params_list[2]),'lambda_2':float(params_list[3]),'fit_intercept':bool(strtobool(params_list[4])),'normalize':bool(strtobool(params_list[5]))};
lrb_m = linear_model.BayesianRidge(**params);

print "LREN_params used:"+ args.lren_params;
params_list  = [x.strip() for x in args.lren_params.split(',')];
params = {'alpha':float(params_list[0]),'l1_ratio':float(params_list[1]),'fit_intercept':bool(strtobool(params_list[2])),'normalize':bool(strtobool(params_list[3])),'positive':bool(strtobool(params_list[4]))};
lren_m = linear_model.ElasticNet(**params);

print "DT_params used:"+ args.dt_params;
params_list  = [x.strip() for x in args.dt_params.split(',')];
#params = {'max_depth':int(params_list[0]),'min_samples_split':int(params_list[1]),'min_samples_leaf':int(params_list[2]),'max_leaf_nodes':params_list[3]};
params = {'max_depth':int(params_list[0]),'min_samples_split':int(params_list[1]),'min_samples_leaf':int(params_list[2])};
dt_m = tree.DecisionTreeRegressor(**params);

print "DT_params 10 used:"+ args.dt_ten_params;
params_list  = [x.strip() for x in args.dt_ten_params.split(',')];
#params = {'max_depth':int(params_list[0]),'min_samples_split':int(params_list[1]),'min_samples_leaf':int(params_list[2]),'max_leaf_nodes':params_list[3]};
params = {'max_depth':int(params_list[0]),'min_samples_split':int(params_list[1]),'min_samples_leaf':int(params_list[2])};
dt_ten_m = tree.DecisionTreeRegressor(**params);

dt_inf_m = tree.DecisionTreeRegressor();

print "GB_params used:"+ args.gb_params;
params_list  = [x.strip() for x in args.gb_params.split(',')];
params = {'learning_rate':float(params_list[0]),'n_estimators':int(params_list[1]),'max_depth':int(params_list[2]),'min_samples_split':int(params_list[3]),'min_samples_leaf':int(params_list[4])};
gb_m = ensemble.GradientBoostingRegressor(**params);

print "RF_params used:"+ args.rf_params;
params_list  = [x.strip() for x in args.rf_params.split(',')];
params = {'n_estimators':int(params_list[0]),'max_depth':int(params_list[1]),'min_samples_split':int(params_list[2]),'min_samples_leaf':int(params_list[3])};
rf_m = ensemble.RandomForestRegressor(**params);

print "----------------------------------------------------------------------------------------";
#Models are instantiated

#Read samples
train_samples = pd.read_csv(args.train_csv);
test_samples = pd.read_csv(args.test_csv);
features = [x.strip() for x in args.features_list.split(',')];
#print "Features here:\n";
#print features;
#print "----------------------------------------------------------------------------------------";
features_print = ['Cycle','Benchmark'];
features_train = train_samples[features];
features_test = test_samples[features];
power_train = train_samples['Power_mW'];
power_test = test_samples['Power_mW'];
cycle_train = train_samples[features_print];
cycle_test = test_samples[features_print];
#cycle_train = cycle_train.reset_index(drop=True, inplace=True);
cycle_test = test_samples[features_print];
#cycle_test = cycle_test.reset_index(drop=True, inplace=True);

#Print out the csvs - feature only, uniq with count of duplicates
if(args.print_duplicate==1):
    features_and_power = [x.strip() for x in args.features_list.split(',')];
    features_and_power.append('Power_mW');
    total_samples = pd.concat([train_samples,test_samples]);
    data_trimmed = total_samples[features_and_power];
    output_csv = args.output + ".used_features.csv";
    data_trimmed.to_csv(output_csv);
    output_csv = args.output + ".used_features.duplicate_count.csv";
    data_with_duplicate_count = data_trimmed.groupby(data_trimmed.columns.tolist(),as_index=False).size()
    data_with_duplicate_count.to_csv(output_csv);

if(args.power_high_dupcnt==1):
    print "----------------------------------------------------------------------------------------"
    data_with_duplicate_count = pd.read_csv(output_csv, sep=",", header=None);
    to_look_at = data_with_duplicate_count.shape[1];
    args.power_high_dupcnt = data_with_duplicate_count.iloc[data_with_duplicate_count[to_look_at-1].idxmax(axis=0, skipna=True),to_look_at-2]

    print "Duplicate power value used: " + str(args.power_high_dupcnt);
    print "----------------------------------------------------------------------------------------"
else:
    print "----------------------------------------------------------------------------------------"
    print "Duplicate power value used: " + str(args.power_high_dupcnt);
    print "----------------------------------------------------------------------------------------"


#Find rows that are in test and train - Gives Memory error
"""
features_and_power_train = train_samples[features_and_power];
features_and_power_test = test_samples[features_and_power];
common_samples = features_and_power_train.merge(features_and_power_test, how='inner', indicator=False);
output_csv = args.output + ".test_found_in_train.csv";
common_samples.to_csv(output_csv);
"""


#print the features
print "Features used";
print "----------------------------------------------------------------------------------------"
str_n = "";
for i in features:
    name=i.split('.');
    length = len(name);
    if(i.find("d_",0,2)==0):
        to_print = "d_"+ name[length-1];
    else:
        to_print = name[length-1];
    str_n = str_n + to_print + ","
print str_n;

#Train - Test Statistics
print "----------------------------------------------------------------------------------------";
print "Statistics of split";
print "----------------------------------------------------------------------------------------";
power_values = power_train;
mean = sum(power_values)/len(power_values);
var_0 = [x-mean for x in power_values];
var_1 = [pow(x,2) for x in var_0];
var = sum(var_1);
#var = sum(pow(x-mean,2) for x in power_values);
std = math.sqrt(var);
std_p = std/mean;
print "Training samples";
print "Average Power : " + str(mean) + "mW";
print "Min Power : " + str(min(power_values)) + "mW";
print "Max Power : " + str(max(power_values)) + "mW";
print "Std dev Power: " + str(std_p) + "%";
print "";
power_values = power_test;
mean = sum(power_values)/len(power_values);
var_0 = [x-mean for x in power_values];
var_1 = [pow(x,2) for x in var_0];
var = sum(var_1);
#var = sum(pow(x-mean,2) for x in power_values);
std = math.sqrt(var);
std_p = std/mean;
print "Test samples";
print "Average Power : " + str(mean) + "mW";
print "Min Power : " + str(min(power_values)) + "mW";print "Max Power : " + str(max(power_values)) + "mW";
print "Max Power : " + str(max(power_values)) + "mW";
print "Std dev Power: " + str(std_p) + "%";

#Find accuracy
lr_m = lr_m.fit(features_train,power_train);
lr_pred = lr_m.predict(features_test);
#lr_pred_df = pd.DataFrame(lr_pred,columns=list('Pred')).reset_index(drop=True,inplace=True)
lr_pred_df = pd.DataFrame(lr_pred);
#print lr_pred_df.shape;
#print "Hi\n";
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".lr_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_lr_m = return_mape(power_test,lr_pred);
maep_nz_lr_m = return_mape_nz(power_test,lr_pred);
mape_lr_m = return_mape_2(power_test,lr_pred);
maxerror_lr_m = return_maxerror(power_test,lr_pred);
avgerror_lr_m = return_avgerror(power_test,lr_pred);
numzero_lr_m = return_numzero(power_test,lr_pred);

lrl_m = lrl_m.fit(features_train,power_train);
lr_l_pred = lrl_m.predict(features_test);
lr_pred_df = pd.DataFrame(lr_l_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".lr_l_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_lr_l_m = return_mape(power_test,lr_l_pred);
maep_nz_lr_l_m = return_mape_nz(power_test,lr_l_pred);
mape_lr_l_m = return_mape_2(power_test,lr_l_pred);
maxerror_lr_l_m = return_maxerror(power_test,lr_l_pred);
avgerror_lr_l_m = return_avgerror(power_test,lr_l_pred);
numzero_lr_l_m = return_numzero(power_test,lr_l_pred);

lrr_m = lrr_m.fit(features_train,power_train);
lr_r_pred = lrr_m.predict(features_test);
lr_pred_df = pd.DataFrame(lr_r_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".lr_r_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_lr_r_m = return_mape(power_test,lr_r_pred);
maep_nz_lr_r_m = return_mape_nz(power_test,lr_r_pred);
mape_lr_r_m = return_mape_2(power_test,lr_r_pred);
maxerror_lr_r_m = return_maxerror(power_test,lr_r_pred);
avgerror_lr_r_m = return_avgerror(power_test,lr_r_pred);
numzero_lr_r_m = return_numzero(power_test,lr_r_pred);


lrb_m = lrb_m.fit(features_train,power_train);
lr_b_pred = lrb_m.predict(features_test);
lr_pred_df = pd.DataFrame(lr_b_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".lr_b_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_lr_b_m = return_mape(power_test,lr_b_pred);
maep_nz_lr_b_m = return_mape_nz(power_test,lr_b_pred);
mape_lr_b_m = return_mape_2(power_test,lr_b_pred);
maxerror_lr_b_m = return_maxerror(power_test,lr_b_pred);
avgerror_lr_b_m = return_avgerror(power_test,lr_b_pred);
numzero_lr_b_m = return_numzero(power_test,lr_b_pred);

lren_m = lren_m.fit(features_train,power_train);
lr_en_pred = lren_m.predict(features_test);
lr_pred_df = pd.DataFrame(lr_en_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".lr_en_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_lr_en_m = return_mape(power_test,lr_en_pred);
maep_nz_lr_en_m = return_mape_nz(power_test,lr_en_pred);
mape_lr_en_m = return_mape_2(power_test,lr_en_pred);
maxerror_lr_en_m = return_maxerror(power_test,lr_en_pred);
avgerror_lr_en_m = return_avgerror(power_test,lr_en_pred);
numzero_lr_en_m = return_numzero(power_test,lr_en_pred);

dt_m = dt_m.fit(features_train,power_train);
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

#Decision tree importance print

#Decision tree importance print
print "----------------------------------------------------------------------------------------";
print "Decision tree importance";
print "----------------------------------------------------------------------------------------";
print dt_m.feature_importances_ ;
print "Some shapes : \nShape of power_test :"+str(power_test.shape)+"\nShape of power pred :"+str(dt_pred.shape)+"\n";
print "----------------------------------------------------------------------------------------";

dt_ten_m = dt_ten_m.fit(features_train,power_train);
dt_ten_pred = dt_ten_m.predict(features_test);
lr_pred_df = pd.DataFrame(dt_ten_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".dt_ten_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_dt_ten_m = return_mape(power_test,dt_ten_pred);
maep_nz_dt_ten_m = return_mape_nz(power_test,dt_ten_pred);
mape_dt_ten_m = return_mape_2(power_test,dt_ten_pred);
maxerror_dt_ten_m = return_maxerror(power_test,dt_ten_pred);
avgerror_dt_ten_m = return_avgerror(power_test,dt_ten_pred);
numzero_dt_ten_m = return_numzero(power_test,dt_ten_pred);
#Saving model
joblibfile = args.output + ".pkl";
joblib.dump(dt_ten_m,joblibfile); 

dt_inf_m = dt_inf_m.fit(features_train,power_train);
dt_inf_pred = dt_inf_m.predict(features_test);
lr_pred_df = pd.DataFrame(dt_inf_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".dt_inf_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_dt_inf_m = return_mape(power_test,dt_inf_pred);
maep_nz_dt_inf_m = return_mape_nz(power_test,dt_inf_pred);
mape_dt_inf_m = return_mape_2(power_test,dt_inf_pred);
maxerror_dt_inf_m = return_maxerror(power_test,dt_inf_pred);
avgerror_dt_inf_m = return_avgerror(power_test,dt_inf_pred);
numzero_dt_inf_m = return_numzero(power_test,dt_inf_pred);


gb_m = gb_m.fit(features_train,power_train);
gb_pred = gb_m.predict(features_test);
lr_pred_df = pd.DataFrame(gb_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".gb_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_gb_m = return_mape(power_test,gb_pred);
maep_nz_gb_m = return_mape_nz(power_test,gb_pred);
mape_gb_m = return_mape_2(power_test,gb_pred);
maxerror_gb_m = return_maxerror(power_test,gb_pred);
avgerror_gb_m = return_avgerror(power_test,gb_pred);
numzero_gb_m = return_numzero(power_test,gb_pred);


rf_m = rf_m.fit(features_train,power_train);
rf_pred = rf_m.predict(features_test);
lr_pred_df = pd.DataFrame(rf_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".rf_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_rf_m = return_mape(power_test,rf_pred);
maep_nz_rf_m = return_mape_nz(power_test,rf_pred);
mape_rf_m = return_mape_2(power_test,rf_pred);
maxerror_rf_m = return_maxerror(power_test,rf_pred);
avgerror_rf_m = return_avgerror(power_test,rf_pred);
numzero_rf_m = return_numzero(power_test,rf_pred);

maep_avg = return_mape(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
maep_nz_avg = return_mape_nz(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
mape_avg = return_mape_2(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
maxerror_avg = return_maxerror(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
avgerror_avg = return_avgerror(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
numzero_avg = return_numzero(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));

maep_min = return_mape(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));
maep_nz_min = return_mape_nz(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));
mape_min = return_mape_2(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));
maxerror_min = return_maxerror(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));
avgerror_min = return_avgerror(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));
numzero_min = return_numzero(power_test,np.repeat(np.amin(power_train),np.shape(power_test)[0]));

maep_p_dup = return_mape(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));
maep_nz_p_dup = return_mape_nz(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));
mape_p_dup = return_mape_2(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));
maxerror_p_dup = return_maxerror(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));
avgerror_p_dup = return_avgerror(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));
numzero_p_dup = return_numzero(power_test,np.repeat(args.power_high_dupcnt,np.shape(power_test)[0]));

#the two mode model - for now I am using the power numbers directly to identify if it's idle (+10% minimum power in the test set) or not.  If it's idle, we give the minimum power else we give the mean power of the block
two_mode_pred = two_mode_m_predict(power_test);
lr_pred_df = pd.DataFrame(two_mode_pred);
concat = pd.concat([cycle_test,lr_pred_df],axis=1)
concat_csv = args.output + ".two_mode_pred.csv";
concat.to_csv(concat_csv,index=False);
maep_two_mode_m = return_mape(power_test,two_mode_pred);
maep_nz_two_mode_m = return_mape_nz(power_test,two_mode_pred);
mape_two_mode_m = return_mape_2(power_test,two_mode_pred);
maxerror_two_mode_m = return_maxerror(power_test,two_mode_pred);
avgerror_two_mode_m = return_avgerror(power_test,two_mode_pred);
numzero_two_mode_m = return_numzero(power_test,two_mode_pred);


print "----------------------------------------------------------------------------------------";
print "Results";
print "----------------------------------------------------------------------------------------";
"""
#Old Print 
print "Model"+ ","  + "Avg" + "," +  "LR"  + "," + "LR_L" + "," + "LR_R"  + "," + "LR_B"  + "," +  "LR_EN" + "," +  "DT" + "," + "DT_ten" + "," + "GB" + "," +  "RF" ;
print "MAEP" + "," + str(maep_avg) + "," + str(maep_lr_m) + "," +  str(maep_lr_l_m) + "," +  str(maep_lr_r_m) + "," +  str(maep_lr_b_m) + "," +  str(maep_lr_en_m) + "," +  str(maep_dt_m) + "," +  str(maep_dt_ten_m) + "," +  str(maep_gb_m) + "," +  str(maep_rf_m) ; 
print "MAPE" + "," + str(mape_avg) + "," + str(mape_lr_m) + "," +  str(mape_lr_l_m) + "," +  str(mape_lr_r_m) + "," +  str(mape_lr_b_m) + "," +  str(mape_lr_en_m) + "," +  str(mape_dt_m) +  "," +  str(mape_dt_ten_m) + "," +  str(mape_gb_m) + "," +  str(mape_rf_m) ;
print "Max_error based on MAEP" + "," + str(maxerror_avg) + "," + str(maxerror_lr_m) + "," +  str(maxerror_lr_l_m) + "," +  str(maxerror_lr_r_m) + "," +  str(maxerror_lr_b_m) + "," +  str(maxerror_lr_en_m) + "," +  str(maxerror_dt_m) +  "," +  str(maxerror_dt_ten_m) + "," +  str(maxerror_gb_m) + "," +  str(maxerror_rf_m) ;
print "Avg_error based on MAEP" + "," + str(avgerror_avg) + "," + str(avgerror_lr_m) + "," +  str(avgerror_lr_l_m) + "," +  str(avgerror_lr_r_m) + "," +  str(avgerror_lr_b_m) + "," +  str(avgerror_lr_en_m) + "," +  str(avgerror_dt_m) +  "," +  str(avgerror_dt_ten_m) + "," +  str(avgerror_gb_m) + "," +  str(avgerror_rf_m) ;
print "No of 0% error in test set of size" + "," + str(numzero_avg) + "," + str(numzero_lr_m) + "," +  str(numzero_lr_l_m) + "," +  str(numzero_lr_r_m) + "," +  str(numzero_lr_b_m) + "," +  str(numzero_lr_en_m) + "," +  str(numzero_dt_m) +  "," +  str(numzero_dt_ten_m) + "," +  str(numzero_gb_m) + "," +  str(numzero_rf_m) ;
print "MAEP on nonzero elements" + "," + str(maep_nz_avg) + "," + str(maep_nz_lr_m) + "," +  str(maep_nz_lr_l_m) + "," +  str(maep_nz_lr_r_m) + "," +  str(maep_nz_lr_b_m) + "," +  str(maep_nz_lr_en_m) + "," +  str(maep_nz_dt_m) + "," +  str(maep_nz_dt_ten_m) + "," +  str(maep_nz_gb_m) + "," +  str(maep_nz_rf_m) ; 
"""
table = [["MAEP" , str(round(maep_avg,2)), str(round(maep_min,2)) , str(round(maep_p_dup,2)),  str(round(maep_two_mode_m,2)) , str(round(maep_lr_m,2))  ,  str(round(maep_lr_l_m,2)) ,  str(round(maep_lr_r_m,2))   ,  str(round(maep_lr_b_m,2)) ,  str(round(maep_lr_en_m,2))   ,  str(round(maep_dt_m,2)) ,  str(round(maep_dt_ten_m,2))  ,  str(round(maep_dt_inf_m,2)) ,  str(round(maep_gb_m,2)) ,  str(round(maep_rf_m,2))]]; 
table.append(["MAPE" , str(round(mape_avg,2)), str(round(mape_min,2)) , str(round(mape_p_dup,2)),  str(round(mape_two_mode_m,2)) , str(round(mape_lr_m,2)) ,  str(round(mape_lr_l_m,2)) ,  str(round(mape_lr_r_m,2)) ,  str(round(mape_lr_b_m,2)) ,  str(round(mape_lr_en_m,2)) ,  str(round(mape_dt_m,2)) ,  str(round(mape_dt_ten_m,2)),  str(round(mape_dt_inf_m,2)) ,  str(round(mape_gb_m,2)) ,  str(round(mape_rf_m,2)) ]);
table.append(["Max_error based on MAEP" , str(round(maxerror_avg[0],2)),  str(round(maxerror_min[0],2)),  str(round(maxerror_p_dup[0],2)),  str(round(maxerror_two_mode_m[0],2)) , str(round(maxerror_lr_m[0],2)) ,  str(round(maxerror_lr_l_m[0],2)) ,  str(round(maxerror_lr_r_m[0],2)) ,  str(round(maxerror_lr_b_m[0],2)) ,  str(round(maxerror_lr_en_m[0],2)) ,  str(round(maxerror_dt_m[0],2))  ,   str(round(maxerror_dt_ten_m[0],2)) ,   str(round(maxerror_dt_inf_m[0],2)),  str(round(maxerror_gb_m[0],2)) ,  str(round(maxerror_rf_m[0],2)) ]);
table.append(["Avg_error based on MAEP" , str(round(avgerror_avg,2)),   str(round(avgerror_min,2)),   str(round(avgerror_p_dup,2)),  str(round(avgerror_two_mode_m,2)) , str(round(avgerror_lr_m,2)) ,  str(round(avgerror_lr_l_m,2)) ,  str(round(avgerror_lr_r_m,2)) ,  str(round(avgerror_lr_b_m,2)) ,  str(round(avgerror_lr_en_m,2)) ,  str(round(avgerror_dt_m,2)) ,  str(round(avgerror_dt_ten_m,2)),  str(round(avgerror_dt_inf_m,2)) ,  str(round(avgerror_gb_m,2)) ,  str(round(avgerror_rf_m,2)) ]);
table.append(["No of 0% error" , str(round(numzero_avg,2)),    str(round(numzero_min,2)),    str(round(numzero_p_dup,2)),  str(round(numzero_two_mode_m,2)) , str(round(numzero_lr_m,2)) ,  str(round(numzero_lr_l_m,2)) ,  str(round(numzero_lr_r_m,2)) ,  str(round(numzero_lr_b_m,2)) ,  str(round(numzero_lr_en_m,2)) ,  str(round(numzero_dt_m,2)) , str(round(numzero_dt_ten_m,2)), str(round(numzero_dt_inf_m,2)) ,  str(round(numzero_gb_m,2)) ,  str(round(numzero_rf_m,2)) ]);
table.append(["MAEP on nonzero elements" , str(round(maep_nz_avg,2)),    str(round(maep_nz_min,2)),    str(round(maep_nz_p_dup,2)),  str(round(maep_nz_two_mode_m,2)) , str(round(maep_nz_lr_m,2)) ,  str(round(maep_nz_lr_l_m,2)) ,  str(round(maep_nz_lr_r_m,2)) ,  str(round(maep_nz_lr_b_m,2)) ,  str(round(maep_nz_lr_en_m,2)) ,  str(round(maep_nz_dt_m,2)) ,  str(round(maep_nz_dt_ten_m,2)),  str(round(maep_nz_dt_inf_m,2)) ,  str(round(maep_nz_gb_m,2)) ,  str(round(maep_nz_rf_m,2)) ]); 
print "\n---------------------------------------------------------------------------------------------------------------------------------------------------";
print(tabulate(table,headers=["Model" , "Avg" , "Pred_min", "Pred_dup", "Two_mode",  "LR" , "LR_L" , "LR_R"  , "LR_B"  ,  "LR_EN" ,  "DT" , "DT_ten" , "DT_inf", "GB" ,  "RF"], tablefmt='orgtbl'));
print "---------------------------------------------------------------------------------------------------------------------------------------------------\n\n";
#Plot histogram of least error model - error

list = [maep_avg, maep_lr_m,  maep_lr_l_m,  maep_lr_r_m,  maep_lr_b_m, maep_lr_en_m,  maep_dt_m,  maep_dt_ten_m,  maep_gb_m,  maep_rf_m];
minpos = list.index(min(list));
maxerr_dt = maxerror_dt_m[0]; 
need_dt = maxerror_dt_m[1]; 
predcited_dt = maxerror_dt_m[2];

if(minpos==0):
    p_buf_pred = np.repeat(np.mean(power_train),np.shape(power_test)[0]);
    maxerr = maxerror_avg[0];
    need = maxerror_avg[1]; 
    predcited = maxerror_avg[2];
    min_err_model = "avg";
elif(minpos==1):
    p_buf_pred = lr_pred; 
    need = maxerror_lr_m[1]; 
    maxerr = maxerror_lr_m[0]; 
    predcited = maxerror_lr_m[2];
    min_err_model = "lr_m";
elif(minpos==2):
    p_buf_pred = lr_l_pred; 
    maxerr = maxerror_lr_m[0]; 
    need = maxerror_lr_m[1]; 
    predcited = maxerror_lr_m[2];
    min_err_model = "lr_l_m";
elif(minpos==3):
    min_err_model = "lr_r_m";
    maxerr = maxerror_lr_m[0]; 
    need = maxerror_lr_m[1]; 
    predcited = maxerror_lr_m[2];
    p_buf_pred = lr_r_pred; 
elif(minpos==4):
    p_buf_pred = lr_b_pred; 
    maxerr = maxerror_lr_b_m[0]; 
    need = maxerror_lr_b_m[1]; 
    predcited = maxerror_lr_b_m[2];
    min_err_model = "lr_b_m";
elif(minpos==5):
    p_buf_pred = lr_en_pred; 
    maxerr = maxerror_lr_en_m[0]; 
    need = maxerror_lr_en_m[1]; 
    predcited = maxerror_lr_en_m[2];
    min_err_model = "lr_en_m";
elif(minpos==6):
    p_buf_pred = dt_pred; 
    maxerr = maxerror_dt_m[0]; 
    need = maxerror_dt_m[1]; 
    predcited = maxerror_dt_m[2];
    min_err_model = "dt_m";
elif(minpos==7):
    p_buf_pred = dt_ten_pred; 
    maxerr = maxerror_dt_ten_m[0]; 
    need = maxerror_dt_ten_m[1]; 
    predcited = maxerror_dt_ten_m[2];
    min_err_model = "dt_ten_m";
elif(minpos==8):
    p_buf_pred = gb_pred; 
    maxerr = maxerror_gb_m[0]; 
    need = maxerror_gb_m[1]; 
    predcited = maxerror_gb_m[2];
    min_err_model = "gb_m";
elif(minpos==9):
    p_buf_pred = rf_pred; 
    maxerr = maxerror_rf_m[0]; 
    need = maxerror_rf_m[1]; 
    predcited = maxerror_rf_m[2];
    min_err_model = "rf_m";
    
#Histogram of power test
counts, bin_edges = np.histogram(power_test,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - Power test";
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";


#Min error model
abs_diff = abs(power_test - p_buf_pred);
error =  (abs_diff)*100/np.mean(power_test);
counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+str(min_err_model);
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";

print "Max error of "+ str(maxerr) +"\nNeed: " + str(need) + "\nPredicted: " + str(predcited);
print "----------------------------------------------------------------------------------------";

print "DT - 20 : Max error of "+ str(maxerr_dt) +"\nNeed: " + str(need_dt) + "\nPredicted: " + str(predcited_dt);
#Average error model
p_buf_pred = np.repeat(np.mean(power_train),np.shape(power_test)[0]);
abs_diff = abs(power_test - p_buf_pred);
error =  (abs_diff)*100/np.mean(power_test);
counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+ "Average_Power_predicting_model"
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";

#Two mode -  error model
p_buf_pred = two_mode_pred;
abs_diff = abs(power_test - p_buf_pred);
error =  (abs_diff)*100/np.mean(power_test);
counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+ "Two mode predicting model";
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";


"""
fig1,ax1 = plt.subplots();
plt.title("args.output"+"min_err_model");
plt.xlabel('Error perentage ');
plt.ylabel('Count');
plt.hist(error);
output_file_name = args.output + ".error_plot.default.jpg";
fig1.savefig(output_file_name,dpi=300,bbox_inches='tight');
"""


### Errors exempting leakage power
def return_idle_exempt_mape_2(p_buf_power_test,p_buf_pred):
    norm_abs_diff = abs(p_buf_power_test - p_buf_pred)/p_buf_power_test;
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(norm_abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    return(np.mean(np.asarray(l))*100);
def return_idle_exempt_mape(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    mean_abs_diff = np.mean(np.asarray(l));
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape

def return_idle_exempt_maxerror(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    mean_abs_diff = np.amax(np.asarray(l));
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return [mape, 0, 0]


def return_idle_exempt_avgerror(p_buf_power_test,p_buf_pred):
    abs_diff = (p_buf_power_test - p_buf_pred);
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    mean_abs_diff = np.mean(np.asarray(l));
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape

def return_idle_exempt_numzero(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    abs_diff = np.asarray(l);
    return (len(abs_diff) - np.count_nonzero(abs_diff));

def return_idle_exempt_mape_nz(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    power_threshold = np.amin(power_test)*1.1;
    l = [];
    iter_cnt = 0;
    for i in np.nditer(power_test):
        if(i>power_threshold):
            l.append(abs_diff[iter_cnt]);
        iter_cnt = iter_cnt+1;
    abs_diff = np.asarray(l);
    mean_abs_diff = sum(abs_diff)/(np.count_nonzero(abs_diff));
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape

#Find accuracy

maep_lr_m = return_idle_exempt_mape(power_test,lr_pred);
maep_nz_lr_m = return_idle_exempt_mape_nz(power_test,lr_pred);
mape_lr_m = return_idle_exempt_mape_2(power_test,lr_pred);
maxerror_lr_m = return_idle_exempt_maxerror(power_test,lr_pred);
avgerror_lr_m = return_idle_exempt_avgerror(power_test,lr_pred);
numzero_lr_m = return_idle_exempt_numzero(power_test,lr_pred);

maep_lr_l_m = return_idle_exempt_mape(power_test,lr_l_pred);
maep_nz_lr_l_m = return_idle_exempt_mape_nz(power_test,lr_l_pred);
mape_lr_l_m = return_idle_exempt_mape_2(power_test,lr_l_pred);
maxerror_lr_l_m = return_idle_exempt_maxerror(power_test,lr_l_pred);
avgerror_lr_l_m = return_idle_exempt_avgerror(power_test,lr_l_pred);
numzero_lr_l_m = return_idle_exempt_numzero(power_test,lr_l_pred);

maep_lr_r_m = return_idle_exempt_mape(power_test,lr_r_pred);
maep_nz_lr_r_m = return_idle_exempt_mape_nz(power_test,lr_r_pred);
mape_lr_r_m = return_idle_exempt_mape_2(power_test,lr_r_pred);
maxerror_lr_r_m = return_idle_exempt_maxerror(power_test,lr_r_pred);
avgerror_lr_r_m = return_idle_exempt_avgerror(power_test,lr_r_pred);
numzero_lr_r_m = return_idle_exempt_numzero(power_test,lr_r_pred);


maep_lr_b_m = return_idle_exempt_mape(power_test,lr_b_pred);
maep_nz_lr_b_m = return_idle_exempt_mape_nz(power_test,lr_b_pred);
mape_lr_b_m = return_idle_exempt_mape_2(power_test,lr_b_pred);
maxerror_lr_b_m = return_idle_exempt_maxerror(power_test,lr_b_pred);
avgerror_lr_b_m = return_idle_exempt_avgerror(power_test,lr_b_pred);
numzero_lr_b_m = return_idle_exempt_numzero(power_test,lr_b_pred);

maep_lr_en_m = return_idle_exempt_mape(power_test,lr_en_pred);
maep_nz_lr_en_m = return_idle_exempt_mape_nz(power_test,lr_en_pred);
mape_lr_en_m = return_idle_exempt_mape_2(power_test,lr_en_pred);
maxerror_lr_en_m = return_idle_exempt_maxerror(power_test,lr_en_pred);
avgerror_lr_en_m = return_idle_exempt_avgerror(power_test,lr_en_pred);
numzero_lr_en_m = return_idle_exempt_numzero(power_test,lr_en_pred);

maep_dt_m = return_idle_exempt_mape(power_test,dt_pred);
maep_nz_dt_m = return_idle_exempt_mape_nz(power_test,dt_pred);
mape_dt_m = return_idle_exempt_mape_2(power_test,dt_pred);
maxerror_dt_m = return_idle_exempt_maxerror(power_test,dt_pred);
avgerror_dt_m = return_idle_exempt_avgerror(power_test,dt_pred);
numzero_dt_m = return_idle_exempt_numzero(power_test,dt_pred);

maep_dt_ten_m = return_idle_exempt_mape(power_test,dt_ten_pred);
maep_nz_dt_ten_m = return_idle_exempt_mape_nz(power_test,dt_ten_pred);
mape_dt_ten_m = return_idle_exempt_mape_2(power_test,dt_ten_pred);
maxerror_dt_ten_m = return_idle_exempt_maxerror(power_test,dt_ten_pred);
avgerror_dt_ten_m = return_idle_exempt_avgerror(power_test,dt_ten_pred);
numzero_dt_ten_m = return_idle_exempt_numzero(power_test,dt_ten_pred);


maep_gb_m = return_idle_exempt_mape(power_test,gb_pred);
maep_nz_gb_m = return_idle_exempt_mape_nz(power_test,gb_pred);
mape_gb_m = return_idle_exempt_mape_2(power_test,gb_pred);
maxerror_gb_m = return_idle_exempt_maxerror(power_test,gb_pred);
avgerror_gb_m = return_idle_exempt_avgerror(power_test,gb_pred);
numzero_gb_m = return_idle_exempt_numzero(power_test,gb_pred);


maep_rf_m = return_idle_exempt_mape(power_test,rf_pred);
maep_nz_rf_m = return_idle_exempt_mape_nz(power_test,rf_pred);
mape_rf_m = return_idle_exempt_mape_2(power_test,rf_pred);
maxerror_rf_m = return_idle_exempt_maxerror(power_test,rf_pred);
avgerror_rf_m = return_idle_exempt_avgerror(power_test,rf_pred);
numzero_rf_m = return_idle_exempt_numzero(power_test,rf_pred);

maep_avg = return_idle_exempt_mape(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
maep_nz_avg = return_idle_exempt_mape_nz(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
mape_avg = return_idle_exempt_mape_2(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
maxerror_avg = return_idle_exempt_maxerror(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
avgerror_avg = return_idle_exempt_avgerror(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));
numzero_avg = return_idle_exempt_numzero(power_test,np.repeat(np.mean(power_train),np.shape(power_test)[0]));

#the two mode model - for now I am using the power numbers directly to identify
#if it's idle (+10% minimum power in the test set) or not.  If it's idle, we
#give the minimum power else we give the mean power of the block (nonidle
#portion)
maep_two_mode_m = return_idle_exempt_mape(power_test,two_mode_pred);
maep_nz_two_mode_m = return_idle_exempt_mape_nz(power_test,two_mode_pred);
mape_two_mode_m = return_idle_exempt_mape_2(power_test,two_mode_pred);
maxerror_two_mode_m = return_idle_exempt_maxerror(power_test,two_mode_pred);
avgerror_two_mode_m = return_idle_exempt_avgerror(power_test,two_mode_pred);
numzero_two_mode_m = return_idle_exempt_numzero(power_test,two_mode_pred);


print "----------------------------------------------------------------------------------------";
print "Results - exempting all idle power";
print "----------------------------------------------------------------------------------------";
"""
#Old Print 
print "Model"+ ","  + "Avg" + "," +  "LR"  + "," + "LR_L" + "," + "LR_R"  + "," + "LR_B"  + "," +  "LR_EN" + "," +  "DT" + "," + "DT_ten" + "," + "GB" + "," +  "RF" ;
print "MAEP" + "," + str(maep_avg) + "," + str(maep_lr_m) + "," +  str(maep_lr_l_m) + "," +  str(maep_lr_r_m) + "," +  str(maep_lr_b_m) + "," +  str(maep_lr_en_m) + "," +  str(maep_dt_m) + "," +  str(maep_dt_ten_m) + "," +  str(maep_gb_m) + "," +  str(maep_rf_m) ; 
print "MAPE" + "," + str(mape_avg) + "," + str(mape_lr_m) + "," +  str(mape_lr_l_m) + "," +  str(mape_lr_r_m) + "," +  str(mape_lr_b_m) + "," +  str(mape_lr_en_m) + "," +  str(mape_dt_m) +  "," +  str(mape_dt_ten_m) + "," +  str(mape_gb_m) + "," +  str(mape_rf_m) ;
print "Max_error based on MAEP" + "," + str(maxerror_avg) + "," + str(maxerror_lr_m) + "," +  str(maxerror_lr_l_m) + "," +  str(maxerror_lr_r_m) + "," +  str(maxerror_lr_b_m) + "," +  str(maxerror_lr_en_m) + "," +  str(maxerror_dt_m) +  "," +  str(maxerror_dt_ten_m) + "," +  str(maxerror_gb_m) + "," +  str(maxerror_rf_m) ;
print "Avg_error based on MAEP" + "," + str(avgerror_avg) + "," + str(avgerror_lr_m) + "," +  str(avgerror_lr_l_m) + "," +  str(avgerror_lr_r_m) + "," +  str(avgerror_lr_b_m) + "," +  str(avgerror_lr_en_m) + "," +  str(avgerror_dt_m) +  "," +  str(avgerror_dt_ten_m) + "," +  str(avgerror_gb_m) + "," +  str(avgerror_rf_m) ;
print "No of 0% error in test set of size" + "," + str(numzero_avg) + "," + str(numzero_lr_m) + "," +  str(numzero_lr_l_m) + "," +  str(numzero_lr_r_m) + "," +  str(numzero_lr_b_m) + "," +  str(numzero_lr_en_m) + "," +  str(numzero_dt_m) +  "," +  str(numzero_dt_ten_m) + "," +  str(numzero_gb_m) + "," +  str(numzero_rf_m) ;
print "MAEP on nonzero elements" + "," + str(maep_nz_avg) + "," + str(maep_nz_lr_m) + "," +  str(maep_nz_lr_l_m) + "," +  str(maep_nz_lr_r_m) + "," +  str(maep_nz_lr_b_m) + "," +  str(maep_nz_lr_en_m) + "," +  str(maep_nz_dt_m) + "," +  str(maep_nz_dt_ten_m) + "," +  str(maep_nz_gb_m) + "," +  str(maep_nz_rf_m) ; 
"""
table = [["MAEP" , str(round(maep_avg,2)),  str(round(maep_two_mode_m,2)) , str(round(maep_lr_m,2))  ,  str(round(maep_lr_l_m,2)) ,  str(round(maep_lr_r_m,2))   ,  str(round(maep_lr_b_m,2)) ,  str(round(maep_lr_en_m,2))   ,  str(round(maep_dt_m,2)) ,  str(round(maep_dt_ten_m,2))   ,  str(round(maep_gb_m,2)) ,  str(round(maep_rf_m,2))]]; 
table.append(["MAPE" , str(round(mape_avg,2)),  str(round(mape_two_mode_m,2)) , str(round(mape_lr_m,2)) ,  str(round(mape_lr_l_m,2)) ,  str(round(mape_lr_r_m,2)) ,  str(round(mape_lr_b_m,2)) ,  str(round(mape_lr_en_m,2)) ,  str(round(mape_dt_m,2)) ,  str(round(mape_dt_ten_m,2)) ,  str(round(mape_gb_m,2)) ,  str(round(mape_rf_m,2)) ]);
table.append(["Max_error based on MAEP" , str(round(maxerror_avg[0],2)),  str(round(maxerror_two_mode_m[0],2)) , str(round(maxerror_lr_m[0],2)) ,  str(round(maxerror_lr_l_m[0],2)) ,  str(round(maxerror_lr_r_m[0],2)) ,  str(round(maxerror_lr_b_m[0],2)) ,  str(round(maxerror_lr_en_m[0],2)) ,  str(round(maxerror_dt_m[0],2))  ,   str(round(maxerror_dt_ten_m[0],2)) ,  str(round(maxerror_gb_m[0],2)) ,  str(round(maxerror_rf_m[0],2)) ]);
table.append(["Avg_error based on MAEP" , str(round(avgerror_avg,2)),  str(round(avgerror_two_mode_m,2)) , str(round(avgerror_lr_m,2)) ,  str(round(avgerror_lr_l_m,2)) ,  str(round(avgerror_lr_r_m,2)) ,  str(round(avgerror_lr_b_m,2)) ,  str(round(avgerror_lr_en_m,2)) ,  str(round(avgerror_dt_m,2)) ,  str(round(avgerror_dt_ten_m,2)) ,  str(round(avgerror_gb_m,2)) ,  str(round(avgerror_rf_m,2)) ]);
table.append(["No of 0% error in test set of size" , str(round(numzero_avg,2)),  str(round(numzero_two_mode_m,2)) , str(round(numzero_lr_m,2)) ,  str(round(numzero_lr_l_m,2)) ,  str(round(numzero_lr_r_m,2)) ,  str(round(numzero_lr_b_m,2)) ,  str(round(numzero_lr_en_m,2)) ,  str(round(numzero_dt_m,2)) , str(round(numzero_dt_ten_m,2)) ,  str(round(numzero_gb_m,2)) ,  str(round(numzero_rf_m,2)) ]);
table.append(["MAEP on nonzero elements" , str(round(maep_nz_avg,2)),  str(round(maep_nz_two_mode_m,2)) , str(round(maep_nz_lr_m,2)) ,  str(round(maep_nz_lr_l_m,2)) ,  str(round(maep_nz_lr_r_m,2)) ,  str(round(maep_nz_lr_b_m,2)) ,  str(round(maep_nz_lr_en_m,2)) ,  str(round(maep_nz_dt_m,2)) ,  str(round(maep_nz_dt_ten_m,2)) ,  str(round(maep_nz_gb_m,2)) ,  str(round(maep_nz_rf_m,2)) ]); 
print "\n-----------------------------------------------------------------------------------------------------------------------------------";
print(tabulate(table,headers=["Model" , "Avg" , "Two_mode",  "LR"  , "LR_L" , "LR_R"  , "LR_B"  ,  "LR_EN" ,  "DT" , "DT_ten" , "GB" ,  "RF"], tablefmt='orgtbl'));
print "-----------------------------------------------------------------------------------------------------------------------------------\n\n";

#Plot histogram of least error model - error

list = [maep_avg, maep_lr_m,  maep_lr_l_m,  maep_lr_r_m,  maep_lr_b_m, maep_lr_en_m,  maep_dt_m,  maep_dt_ten_m,  maep_gb_m,  maep_rf_m];
minpos = list.index(min(list));
if(minpos==0):
    p_buf_pred = np.repeat(np.mean(power_train),np.shape(power_test)[0]);
    min_err_model = "avg";
elif(minpos==1):
    p_buf_pred = lr_pred; 
    min_err_model = "lr_m";
elif(minpos==2):
    p_buf_pred = lr_l_pred; 
    min_err_model = "lr_l_m";
elif(minpos==3):
    min_err_model = "lr_r_m";
    p_buf_pred = lr_r_pred; 
elif(minpos==4):
    p_buf_pred = lr_b_pred; 
    min_err_model = "lr_b_m";
elif(minpos==5):
    p_buf_pred = lr_en_pred; 
    min_err_model = "lr_en_m";
elif(minpos==6):
    p_buf_pred = dt_pred; 
    min_err_model = "dt_m";
elif(minpos==7):
    p_buf_pred = dt_ten_pred; 
    min_err_model = "dt_ten_m";
elif(minpos==8):
    p_buf_pred = gb_pred; 
    min_err_model = "gb_m";
elif(minpos==9):
    p_buf_pred = rf_pred; 
    min_err_model = "rf_m";
    
#Min error model
abs_diff = abs(power_test - p_buf_pred);
power_threshold = np.amin(power_test)*1.1;
l = [];
m = [];
iter_cnt = 0;

for i in np.nditer(power_test):
    if(i>power_threshold):
        l.append(abs_diff[iter_cnt]);
        m.append(power_test[iter_cnt]);
    iter_cnt = iter_cnt+1;
abs_diff = (np.asarray(l));
power_test_short = (np.asarray(m));
#Histogram of power test
counts, bin_edges = np.histogram(power_test_short,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - Power test short"+ "|| Total samples :" + str(len(power_test_short));
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";


error =  (abs_diff)*100/np.mean(power_test);




counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+str(min_err_model) + "|| Total samples :" + str(len(abs_diff));
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";

#Average error model
p_buf_pred = np.repeat(np.mean(power_train),np.shape(power_test)[0]);
abs_diff = abs(power_test - p_buf_pred);
power_threshold = np.amin(power_test)*1.1;
l = [];
iter_cnt = 0;

for i in np.nditer(power_test):
    if(i>power_threshold):
        l.append(abs_diff[iter_cnt]);
    iter_cnt = iter_cnt+1;
abs_diff = (np.asarray(l));
error =  (abs_diff)*100/np.mean(power_test);
counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+ "Average_Power_predicting_model"
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";

#Two mode -  error model
p_buf_pred = two_mode_pred;
abs_diff = abs(power_test - p_buf_pred);
power_threshold = np.amin(power_test)*1.1;
l = [];
iter_cnt = 0;

for i in np.nditer(power_test):
    if(i>power_threshold):
        l.append(abs_diff[iter_cnt]);
    iter_cnt = iter_cnt+1;
abs_diff = (np.asarray(l));
error =  (abs_diff)*100/np.mean(power_test);
counts, bin_edges = np.histogram(error,bins=20);
print "----------------------------------------------------------------------------------------";
print "Histogram - "+ "Two mode predicting model";
print "----------------------------------------------------------------------------------------";
#table = [["Bins","Count"]];
table = [];
for i in range(counts.shape[0]):
  #print str(round(bin_edges[i],2)) + " to " + str(round(bin_edges[i+1],2)) + "    |             " + str(counts[i]);
  table.append([(str(round(bin_edges[i],2)) + " to  " + str(round(bin_edges[i+1],2))), str(counts[i])]);

print(tabulate(table,headers=["Bins","Count"], tablefmt='orgtbl'));
print "----------------------------------------------------------------------------------------";



