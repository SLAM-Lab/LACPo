#!/usr/bin/env python2
# -*- coding: utf-8 -*-
###Replace <FEATURE_INDEX> with index of the corresponding blocks features in the csv

#import pickle;
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
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import math;
import graphviz;


def error_metric(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    diff = (p_buf_power_test - p_buf_pred);
    print "Absolute difference:";
    print "--------------------";
    print "Min Abs difference :"+str(min(abs_diff));
    print "Max Abs diffrence:"+str(max(abs_diff));
    mean_abs_diff = sum(abs_diff)/len(abs_diff);
    print "Mean abs error :"+str(mean_abs_diff);
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    print "MAPE="+str(mape)+"%";
    print "Difference";
    print "--------------------";
    print "Min difference :"+str(min(diff));
    print "Max diffrence:"+str(max(diff));
    mean_diff = sum(diff)/len(diff);
    print "Mean error :"+str(mean_diff);
    mape =  (mean_diff)*100/np.mean(p_buf_power_test);
    print "MPE="+str(mape)+"%";
 #   print "Root mean squared error :"+ str(math.sqrt(mean_squared_error(p_buf_power_test,p_buf_pred)));
    print "---------------------------------------------------------------";
def return_mape(p_buf_power_test,p_buf_pred):
    abs_diff = abs(p_buf_power_test - p_buf_pred);
    mean_abs_diff = sum(abs_diff)/len(abs_diff);
    mape =  (mean_abs_diff)*100/np.mean(p_buf_power_test);
    return mape


def print_compare(p_buf_power_test,p_buf_pred):
    i=0;
    while(i<p_buf_power_test.size):
        i=i+1;
        print str(p_buf_power_test[i])+","+str(p_buf_pred[i])


def return_mape_2(p_buf_power_test,p_buf_pred):
    norm_abs_diff = abs(p_buf_power_test - p_buf_pred)/p_buf_power_test;
    mape_2 =  np.mean((norm_abs_diff))*100;
    return mape_2

def all_model(mult_features_add,mult_power,dont_split_the_first_data,test_features,test_power):
    if(dont_split_the_first_data):
        p_buf_features_train = mult_features_add;
        p_buf_features_test = test_features;
        p_buf_power_train = mult_power;
        p_buf_power_test = test_power;
    else:
        p_buf_features_train, p_buf_features_test, p_buf_power_train, p_buf_power_test = train_test_split(mult_features_add, mult_power, random_state=5);
    print "Train/Test vetor split:"+str(len(p_buf_features_train))+"/"+str(len(p_buf_features_test));

    print "Min value in the test set:"+str(min(p_buf_power_test));
    print "Max value in the test set :"+str(max(p_buf_power_test));
    print "Average value in the test set :"+str(np.mean(p_buf_power_test));
    
    print "Min value in the train set:"+str(min(p_buf_power_train));
    print "Max value in the train set :"+str(max(p_buf_power_train));
    print "Average value in the train set :"+str(np.mean(p_buf_power_train));
    #print "---------------------------------------------------------------";
    print "\n"


    p_buf_lm = linear_model.LinearRegression(normalize=True);
    p_buf_model = p_buf_lm.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = p_buf_lm.predict(p_buf_features_test);
    mape_linear_model = return_mape(p_buf_power_test,p_buf_pred);
    #print_compare(p_buf_power_test,p_buf_pred);
    #print p_buf_lm.coef_;
    #print p_buf_lm.intercept_;
    mape_linear_model_2 = return_mape_2(p_buf_power_test,p_buf_pred);
    
    p_buf_lm = linear_model.Ridge(alpha=0.1);
    p_buf_model = p_buf_lm.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = p_buf_lm.predict(p_buf_features_test);
    mape_linear_model_ridge=return_mape(p_buf_power_test,p_buf_pred);
    mape_linear_model_ridge_2=return_mape_2(p_buf_power_test,p_buf_pred);

    p_buf_lm = linear_model.Lasso(alpha=0.01);
    p_buf_model = p_buf_lm.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = p_buf_lm.predict(p_buf_features_test);
    mape_linear_model_lasso=return_mape(p_buf_power_test,p_buf_pred);
    mape_linear_model_lasso_2=return_mape_2(p_buf_power_test,p_buf_pred);
    
    p_buf_lm = linear_model.BayesianRidge();
    p_buf_model = p_buf_lm.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = p_buf_lm.predict(p_buf_features_test);
    mape_linear_bayesianridge=return_mape(p_buf_power_test,p_buf_pred);
    mape_linear_bayesianridge_2=return_mape_2(p_buf_power_test,p_buf_pred);
    
    p_buf_lm = tree.DecisionTreeRegressor();
    p_buf_model = p_buf_lm.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = p_buf_lm.predict(p_buf_features_test);
    print "Feature_importances:";
    print p_buf_lm.feature_importances_;
    mape_decision_tree=return_mape(p_buf_power_test,p_buf_pred);
    mape_decision_tree_2=return_mape_2(p_buf_power_test,p_buf_pred);
    

    params = {'n_estimators': 5, 'max_depth': 4, 'min_samples_split': 2,  'learning_rate': 0.01, 'loss': 'ls'};
    clf = ensemble.GradientBoostingRegressor(**params);
    p_buf_model = clf.fit(p_buf_features_train,p_buf_power_train);
    p_buf_pred = clf.predict(p_buf_features_test);
    mape_gradient_boosting=return_mape(p_buf_power_test,p_buf_pred);
    mape_gradient_boosting_2=return_mape_2(p_buf_power_test,p_buf_pred);
             
    mape_average=return_mape(p_buf_power_test,[p_buf_power_test.mean()] * len(p_buf_power_test) );
    mape_average_2=return_mape_2(p_buf_power_test,[p_buf_power_test.mean()] * len(p_buf_power_test) );
    print "Normalized Linear model,"+str(mape_linear_model)+","+str(mape_linear_model_2)
    print "Linear model- Ridge,"+str(mape_linear_model_ridge)+","+str(mape_linear_model_ridge_2)
    print "Linear model- Lasso,"+str(mape_linear_model_lasso)+","+str(mape_linear_model_lasso_2)
    print "Linear model- BayesianRidge,"+str(mape_linear_bayesianridge)+","+str(mape_linear_bayesianridge_2)
    print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)
    print "Gradient Boosting,"+str(mape_gradient_boosting)+","+str(mape_gradient_boosting_2)
    print "Model predicting average number always,"+str(mape_average)+","+str(mape_average_2)

def decomposed_model(if_features_train,if_features_test,if_power_train,if_power_test, id_features_train,id_features_test,id_power_train,id_power_test, ex_features_train,ex_features_test,ex_power_train,ex_power_test, lsu_features_train,lsu_features_test,lsu_power_train,lsu_power_test, csr_features_train,csr_features_test,csr_power_train,csr_power_test, pmp_features_train,pmp_features_test,pmp_power_train,pmp_power_test, top_power_train,top_power_test):
 
    print "Train/Test vetor split:"+str(len(if_features_train))+"/"+str(len(if_features_test));

    print "Min value in the train set:"+str(min(top_power_train));
    print "Max value in the train set :"+str(max(top_power_train));
    print "Average value in the train set :"+str(np.mean(top_power_train));
    
    print "Min value in the test set:"+str(min(top_power_test));
    print "Max value in the test set :"+str(max(top_power_test));
    print "Average value in the test set :"+str(np.mean(top_power_test));
    #print "---------------------------------------------------------------";
    print "\n"
    p_buf_features_train_1 = if_features_train;
    p_buf_features_test_1 = if_features_test;
    p_buf_features_train_2 = id_features_train;
    p_buf_features_test_2 = id_features_test;
    
    p_buf_features_train_3 = ex_features_train;
    p_buf_features_test_3 = ex_features_test;
    p_buf_features_train_4 = lsu_features_train;
    p_buf_features_test_4 = lsu_features_test;
    
    p_buf_features_train_5 = csr_features_train;
    p_buf_features_test_5 = csr_features_test;
    p_buf_features_train_6 = pmp_features_train;
    p_buf_features_test_6 = pmp_features_test;
    
    p_buf_power_train_1 = if_power_train;
    p_buf_power_train_2 = id_power_train;
    p_buf_power_train_3 = ex_power_train;
    p_buf_power_train_4 = lsu_power_train;
    p_buf_power_train_5 = csr_power_train;
    p_buf_power_train_6 = pmp_power_train;
    
    ##Added for diff power
    p_buf_power_train_7 = top_power_train - (if_power_train+id_power_train+ex_power_train+lsu_power_train+csr_power_train+pmp_power_train);
    p_buf_features_train_7 = np.concatenate((np.concatenate((np.concatenate((if_features_train,id_features_train),axis=1), np.concatenate((ex_features_train,lsu_features_train),axis=1)),axis=1),np.concatenate((csr_features_train,pmp_features_train),axis=1)),axis=1);
    p_buf_features_test_7 = np.concatenate((np.concatenate((np.concatenate((if_features_test,id_features_test),axis=1), np.concatenate((ex_features_test,lsu_features_test),axis=1)),axis=1),np.concatenate((csr_features_test,pmp_features_test),axis=1)),axis=1);

    


    p_buf_lm_1 = linear_model.LinearRegression(normalize=True);
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    p_buf_lm_2 = linear_model.LinearRegression(normalize=True);
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    p_buf_lm_3 = linear_model.LinearRegression(normalize=True);
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    p_buf_lm_4 = linear_model.LinearRegression(normalize=True);
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    p_buf_lm_5 = linear_model.LinearRegression(normalize=True);
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    p_buf_lm_6 = linear_model.LinearRegression(normalize=True);
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    ##Added for diff power
    p_buf_lm_7 = linear_model.LinearRegression(normalize=True);
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_linear_model = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_linear_model_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    #mape_linear_model = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_linear_model_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));

    p_buf_lm_1 = linear_model.Ridge(alpha=0.1);
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    p_buf_lm_2 = linear_model.Ridge(alpha=0.1);
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    p_buf_lm_3 = linear_model.Ridge(alpha=0.1);
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    p_buf_lm_4 = linear_model.Ridge(alpha=0.1);
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    p_buf_lm_5 = linear_model.Ridge(alpha=0.1);
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    p_buf_lm_6 = linear_model.Ridge(alpha=0.1);
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    ##Added for diff power
    p_buf_lm_7 = linear_model.Ridge(alpha=0.1);
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_linear_model_ridge = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_linear_model_ridge_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));

    #mape_linear_model_ridge = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_linear_model_ridge_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));

    p_buf_lm_1 = linear_model.Lasso(alpha=0.01);
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    p_buf_lm_2 = linear_model.Lasso(alpha=0.01);
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    p_buf_lm_3 = linear_model.Lasso(alpha=0.01);
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    p_buf_lm_4 = linear_model.Lasso(alpha=0.01);
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    p_buf_lm_5 = linear_model.Lasso(alpha=0.01);
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    p_buf_lm_6 = linear_model.Lasso(alpha=0.01);
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    ##Added for diff power
    p_buf_lm_7 = linear_model.Lasso(alpha=0.01);
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_linear_model_lasso = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_linear_model_lasso_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    
    #mape_linear_model_lasso = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_linear_model_lasso_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    
    
    p_buf_lm_1 = linear_model.BayesianRidge();
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    p_buf_lm_2 = linear_model.BayesianRidge();
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    p_buf_lm_3 = linear_model.BayesianRidge();
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    p_buf_lm_4 = linear_model.BayesianRidge();
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    p_buf_lm_5 = linear_model.BayesianRidge();
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    p_buf_lm_6 = linear_model.BayesianRidge();
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    ##Added for diff power
    p_buf_lm_7 = linear_model.BayesianRidge();
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_linear_bayesianridge = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_linear_bayesianridge_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    
    #mape_linear_bayesianridge = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_linear_bayesianridge_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    
    
    p_buf_lm_1 = tree.DecisionTreeRegressor();
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    #print "Feature_importances 1:";
    #print p_buf_lm_1.feature_importances_
    mape_decision_tree = return_mape(if_power_test,p_buf_pred_1);
    mape_decision_tree_2 = return_mape_2(if_power_test,p_buf_pred_1);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)
    
    p_buf_lm_2 = tree.DecisionTreeRegressor();
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    #print "Feature_importances 2:";
    #print p_buf_lm_2.feature_importances_
    mape_decision_tree = return_mape(id_power_test,p_buf_pred_2);
    mape_decision_tree_2 = return_mape_2(id_power_test,p_buf_pred_2);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)

    p_buf_lm_3 = tree.DecisionTreeRegressor();
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    #print "Feature_importances 3:";
    #print p_buf_lm_3.feature_importances_
    mape_decision_tree = return_mape(ex_power_test,p_buf_pred_3);
    mape_decision_tree_2 = return_mape_2(ex_power_test,p_buf_pred_3);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)

    p_buf_lm_4 = tree.DecisionTreeRegressor();
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    #print "Feature_importances 4:";
    #print p_buf_lm_4.feature_importances_
    mape_decision_tree = return_mape(lsu_power_test,p_buf_pred_4);
    mape_decision_tree_2 = return_mape_2(lsu_power_test,p_buf_pred_4);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)

    p_buf_lm_5 = tree.DecisionTreeRegressor();
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    #print "Feature_importances 5:";
    #print p_buf_lm_5.feature_importances_
    mape_decision_tree = return_mape(csr_power_test,p_buf_pred_5);
    mape_decision_tree_2 = return_mape_2(csr_power_test,p_buf_pred_5);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)

    p_buf_lm_6 = tree.DecisionTreeRegressor();
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    #print "Feature_importances 6:";
    #print p_buf_lm_6.feature_importances_
    mape_decision_tree = return_mape(pmp_power_test,p_buf_pred_6);
    mape_decision_tree_2 = return_mape_2(pmp_power_test,p_buf_pred_6);
    #print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)

    ##Added for diff power
    p_buf_lm_7 = tree.DecisionTreeRegressor();
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_decision_tree = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_decision_tree_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));

    #mape_decision_tree = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_decision_tree_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    
    params = {'n_estimators': 5, 'max_depth': 4, 'min_samples_split': 2,  'learning_rate': 0.01, 'loss': 'ls'};


    p_buf_lm_1 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_1 = p_buf_lm_1.fit(p_buf_features_train_1,p_buf_power_train_1);
    p_buf_pred_1 = p_buf_lm_1.predict(p_buf_features_test_1);
    p_buf_lm_2 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_2 = p_buf_lm_2.fit(p_buf_features_train_2,p_buf_power_train_2);
    p_buf_pred_2 = p_buf_lm_2.predict(p_buf_features_test_2);
    p_buf_lm_3 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_3 = p_buf_lm_3.fit(p_buf_features_train_3,p_buf_power_train_3);
    p_buf_pred_3 = p_buf_lm_3.predict(p_buf_features_test_3);
    p_buf_lm_4 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_4 = p_buf_lm_4.fit(p_buf_features_train_4,p_buf_power_train_4);
    p_buf_pred_4 = p_buf_lm_4.predict(p_buf_features_test_4);
    p_buf_lm_5 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_5 = p_buf_lm_5.fit(p_buf_features_train_5,p_buf_power_train_5);
    p_buf_pred_5 = p_buf_lm_5.predict(p_buf_features_test_5);
    p_buf_lm_6 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_6 = p_buf_lm_6.fit(p_buf_features_train_6,p_buf_power_train_6);
    p_buf_pred_6 = p_buf_lm_6.predict(p_buf_features_test_6);
    ##Added for diff power
    p_buf_lm_7 = ensemble.GradientBoostingRegressor(**params);
    p_buf_model_7 = p_buf_lm_7.fit(p_buf_features_train_7,p_buf_power_train_7);
    p_buf_pred_7 = p_buf_lm_7.predict(p_buf_features_test_7);
    mape_gradient_boosting = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    mape_gradient_boosting_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6+p_buf_pred_7));
    
    #mape_gradient_boosting = return_mape(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    #mape_gradient_boosting_2 = return_mape_2(top_power_test,(p_buf_pred_1+p_buf_pred_2+p_buf_pred_3+p_buf_pred_4+p_buf_pred_5+p_buf_pred_6));
    
              
    mape_average=return_mape(top_power_test,[top_power_train.mean()] * len(top_power_test) );
    mape_average_2=return_mape_2(top_power_test,[top_power_train.mean()] * len(top_power_test) );

    print "Normalized Linear model,"+str(mape_linear_model)+","+str(mape_linear_model_2)
    print "Linear model- Ridge,"+str(mape_linear_model_ridge)+","+str(mape_linear_model_ridge_2)
    print "Linear model- Lasso,"+str(mape_linear_model_lasso)+","+str(mape_linear_model_lasso_2)
    print "Linear model- BayesianRidge,"+str(mape_linear_bayesianridge)+","+str(mape_linear_bayesianridge_2)
    print "Decision Tree,"+str(mape_decision_tree)+","+str(mape_decision_tree_2)
    print "Gradient Boosting,"+str(mape_gradient_boosting)+","+str(mape_gradient_boosting_2)
    print "Model predicting average number always,"+str(mape_average)+","+str(mape_average_2)

    

path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/";
suffix = ".top.parsed.csv"
aes_cbc = path + "aes_cbc" + suffix
conv2d = path + "conv2d" + suffix
fdctfst = path + "fdctfst" + suffix
fft = path + "fft" + suffix
fir = path + "fir" + suffix
matmul = path + "matmul" + suffix
keccak = path + "keccak" + suffix
sha = path + "sha" + suffix
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
#test_features_7 = genfromtxt(sha, delimiter=',');

aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));



test_features = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features = np.vstack((test_features,test_features_2[2:-1,:]));
test_features = np.vstack((test_features,test_features_3[2:-1,:]));
test_features = np.vstack((test_features,test_features_4[2:-1,:]));
test_features = np.vstack((test_features,test_features_5[2:-1,:]));
test_features = np.vstack((test_features,test_features_6[2:-1,:]));

power = test_features[:,-1];
test_features_top_power = power*1000;
print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";

##if_stage
print "Instruction Fetch stage \n";
path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_if_stage/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_if = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features_if = np.vstack((test_features_if,test_features_2[2:-1,:]));
test_features_if = np.vstack((test_features_if,test_features_3[2:-1,:]));
test_features_if = np.vstack((test_features_if,test_features_4[2:-1,:]));
test_features_if = np.vstack((test_features_if,test_features_5[2:-1,:]));
test_features_if = np.vstack((test_features_if,test_features_6[2:-1,:]));

power = test_features_if[:,-1];
test_features_if_power = power*1000;
test_features_if_features = test_features_if[:,[<FEATURE_INDEX>]]; 
print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";
###Features : 1,2,4,5

##id_stage
print "Instruction Decode stage \n";
path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_id_stage/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_id = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features_id = np.vstack((test_features_id,test_features_2[2:-1,:]));
test_features_id = np.vstack((test_features_id,test_features_3[2:-1,:]));
test_features_id = np.vstack((test_features_id,test_features_4[2:-1,:]));
test_features_id = np.vstack((test_features_id,test_features_5[2:-1,:]));
test_features_id = np.vstack((test_features_id,test_features_6[2:-1,:]));
#test_features = np.vstack((test_features,test_features_7[2:-1,0:112]));

power = test_features_id[:,-1];
test_features_id_power = power*1000;
test_features_id_features = test_features_id[:,[<FEATURE_INDEX>]];
print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";

###Features : 1,2,3,4,5,6,20,21,22,23,24,25,26,27,28,30,31,32

##ex_stage
print "Instruction Execution stage \n";

path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_ex_stage/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,110];
conv2d_power = test_features_1[2:-1,110];
fdctfst_power = test_features_2[2:-1,110];
fft_power = test_features_3[2:-1,110];
fir_power = test_features_4[2:-1,110];
matmul_power = test_features_5[2:-1,110];
keccak_power = test_features_6[2:-1,110];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_ex = np.vstack((test_features_0[2:-1,0:112],test_features_1[2:-1,0:112]));
test_features_ex = np.vstack((test_features_ex,test_features_2[2:-1,0:112]));
test_features_ex = np.vstack((test_features_ex,test_features_3[2:-1,0:112]));
test_features_ex = np.vstack((test_features_ex,test_features_4[2:-1,0:112]));
test_features_ex = np.vstack((test_features_ex,test_features_5[2:-1,0:112]));
test_features_ex = np.vstack((test_features_ex,test_features_6[2:-1,0:112]));
#test_features = np.vstack((test_features,test_features_7[2:-1,0:112]));
power = test_features_ex[:,110];
test_features_ex_power = power*1000;
test_features_ex_features = test_features_ex[:,[<FEATURE_INDEX>]];

print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";

###Features : 54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,90,91,92,93,94,95,96,97,98,99,100,101,102,104



##lsu_stage
print "Instruction LSU stage \n";

path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_lsu_input/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_lsu = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features_lsu = np.vstack((test_features_lsu,test_features_2[2:-1,:]));
test_features_lsu = np.vstack((test_features_lsu,test_features_3[2:-1,:]));
test_features_lsu = np.vstack((test_features_lsu,test_features_4[2:-1,:]));
test_features_lsu = np.vstack((test_features_lsu,test_features_5[2:-1,:]));
test_features_lsu = np.vstack((test_features_lsu,test_features_6[2:-1,:]));
power = test_features_lsu[:,-1];
test_features_lsu_power = power*1000;
test_features_lsu_features = test_features_lsu[:,[<FEATURE_INDEX>]];

print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";
##Features : 28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43

##cs_registers_i
print "Instruction CSR stage \n";

path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_cs_registers_i/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
#test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_csr = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features_csr = np.vstack((test_features_csr,test_features_2[2:-1,:]));
test_features_csr = np.vstack((test_features_csr,test_features_3[2:-1,:]));
test_features_csr = np.vstack((test_features_csr,test_features_4[2:-1,:]));
test_features_csr = np.vstack((test_features_csr,test_features_5[2:-1,:]));
test_features_csr = np.vstack((test_features_csr,test_features_6[2:-1,:]));
#test_features = np.vstack((test_features,test_features_7[2:-1,0:112]));
power = test_features_csr[:,-1];
test_features_csr_power = power*1000;
test_features_csr_features = test_features_csr[:,[<FEATURE_INDEX>]];
print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";
##Features : 5,6,7

##pmp_unit_i
print "Instruction PMP stage \n";

path = "/home/local/supreme/ajaykrishna1111/gate_ri5cy_vcds_single_iter/Cumulative_csvs_for_top/Cumulative_csvs_pmp_unit/";
aes_cbc = path + "aes_cbc.cumulative.csv";
conv2d = path + "conv2d.cumulative.csv";
fdctfst = path + "fdctfst.cumulative.csv";
fft = path + "fft.cumulative.csv";
fir = path + "fir.cumulative.csv";
matmul = path + "matmul.cumulative.csv";
keccak = path + "keccak.cumulative.csv";
sha = path + "sha.cumulative.csv";
test_features_0 = genfromtxt(aes_cbc, delimiter=',');
test_features_1 = genfromtxt(conv2d, delimiter=',');
test_features_2 = genfromtxt(fdctfst, delimiter=',');
test_features_3 = genfromtxt(fft, delimiter=',');
test_features_4 = genfromtxt(fir, delimiter=',');
test_features_5 = genfromtxt(matmul, delimiter=',');
test_features_6 = genfromtxt(keccak, delimiter=',');
#test_features_7 = genfromtxt(sha, delimiter=',');
aes_cbc_power = test_features_0[2:-1,-1];
conv2d_power = test_features_1[2:-1,-1];
fdctfst_power = test_features_2[2:-1,-1];
fft_power = test_features_3[2:-1,-1];
fir_power = test_features_4[2:-1,-1];
matmul_power = test_features_5[2:-1,-1];
keccak_power = test_features_6[2:-1,-1];
#sha_power = test_features_7[2:-1,-1];
## Top Power statistics
print "Mean, Standard deviation, max, min";
print "Power of aes_cbc: " + str(np.mean(aes_cbc_power)*1000) + "," + str(np.std(aes_cbc_power)*1000) + "," + str(np.max(aes_cbc_power*1000)) + "," + str(np.min(aes_cbc_power*1000)); 
print "Power of conv2d: " + str(np.mean(conv2d_power)*1000) + "," + str(np.std(conv2d_power)*1000) + "," + str(np.max(conv2d_power*1000)) + "," + str(np.min(conv2d_power*1000));
print "Power of fdctfst: " + str(np.mean(fdctfst_power)*1000) + "," + str(np.std(fdctfst_power)*1000) + "," + str(np.max(fdctfst_power*1000)) + "," + str(np.min(fdctfst_power*1000));
print "Power of fft: " + str(np.mean(fft_power)*1000) + "," + str(np.std(fft_power)*1000) + "," + str(np.max(fft_power*1000)) + "," + str(np.min(fft_power*1000));
print "Power of fir: " + str(np.mean(fir_power)*1000) + "," + str(np.std(fir_power)*1000) + "," + str(np.max(fir_power*1000)) + "," + str(np.min(fir_power*1000));
print "Power of matmul: " + str(np.mean(matmul_power)*1000) + "," + str(np.std(matmul_power)*1000)+ "," + str(np.max(matmul_power*1000)) + "," + str(np.min(matmul_power*1000));
print "Power of keccak: " + str(np.mean(keccak_power)*1000) + "," + str(np.std(keccak_power)*1000)+ "," + str(np.max(keccak_power*1000)) + "," + str(np.min(keccak_power*1000));
#print "Mean Power of sha" + str(np.mean(sha_power)*1000) + "," + str(np.std(sha_power));

test_features_pmp = np.vstack((test_features_0[2:-1,:],test_features_1[2:-1,:]));
test_features_pmp = np.vstack((test_features_pmp,test_features_2[2:-1,:]));
test_features_pmp = np.vstack((test_features_pmp,test_features_3[2:-1,:]));
test_features_pmp = np.vstack((test_features_pmp,test_features_4[2:-1,:]));
test_features_pmp = np.vstack((test_features_pmp,test_features_5[2:-1,:]));
test_features_pmp = np.vstack((test_features_pmp,test_features_6[2:-1,:]));
#test_features = np.vstack((test_features,test_features_7[2:-1,0:112]));
power = test_features_pmp[:,-1];
test_features_pmp_power = power*1000;
test_features_pmp_features = test_features_pmp[:,[<FEATURE_INDEX>]];
print "\n";
print "Power of total: " + str(np.mean(power)*1000) + "," + str(np.std(power)*1000)+ "," + str(np.max(power*1000)) + "," + str(np.min(power*1000));
print "\n";
##Features : 4,5


## K-Fold

kf = KFold(n_splits=10)
kf.get_n_splits(test_features_if)
iter=0;
for train_index, test_index in kf.split(test_features_if):
    print "Iter no:"+str(iter)
    iter=iter+1
    #if(iter==3):
    if_features_train = test_features_if_features[train_index];
    if_features_test = test_features_if_features[test_index];
    if_power_train = test_features_if_power[train_index];
    if_power_test = test_features_if_power[test_index];
    id_features_train = test_features_id_features[train_index];
    id_features_test = test_features_id_features[test_index];
    id_power_train = test_features_id_power[train_index];
    id_power_test = test_features_id_power[test_index];
    ex_features_train = test_features_ex_features[train_index];
    ex_features_test = test_features_ex_features[test_index];
    ex_power_train = test_features_ex_power[train_index];
    ex_power_test = test_features_ex_power[test_index];
    lsu_features_train = test_features_lsu_features[train_index];
    lsu_features_test = test_features_lsu_features[test_index];
    lsu_power_train = test_features_lsu_power[train_index];
    lsu_power_test = test_features_lsu_power[test_index];
    csr_features_train = test_features_csr_features[train_index];
    csr_features_test = test_features_csr_features[test_index];
    csr_power_train  = test_features_csr_power[train_index];
    csr_power_test =  test_features_csr_power[test_index];
    pmp_features_train = test_features_pmp_features[train_index];
    pmp_features_test = test_features_pmp_features[test_index];
    pmp_power_train = test_features_pmp_power[train_index];
    pmp_power_test = test_features_pmp_power[test_index];
    top_power_train = test_features_top_power[train_index];
    top_power_test = test_features_top_power[test_index];
    diff_power_train  = top_power_train - (if_power_train+id_power_train+ex_power_train+lsu_power_train+csr_power_train+pmp_power_train);
    diff_power_test =  top_power_test - (if_power_test+id_power_test+ex_power_test+lsu_power_test+csr_power_test+pmp_power_test);
    features_concat_train = np.concatenate((np.concatenate((np.concatenate((if_features_train,id_features_train),axis=1), np.concatenate((ex_features_train,lsu_features_train),axis=1)),axis=1),np.concatenate((csr_features_train,pmp_features_train),axis=1)),axis=1);
    features_concat_test = np.concatenate((np.concatenate((np.concatenate((if_features_test,id_features_test),axis=1), np.concatenate((ex_features_test,lsu_features_test),axis=1)),axis=1),np.concatenate((csr_features_test,pmp_features_test),axis=1)),axis=1);
     
    decomposed_model(if_features_train,if_features_test,if_power_train,if_power_test, id_features_train,id_features_test,id_power_train,id_power_test, ex_features_train,ex_features_test,ex_power_train,ex_power_test, lsu_features_train,lsu_features_test,lsu_power_train,lsu_power_test, csr_features_train,csr_features_test,csr_power_train,csr_power_test, pmp_features_train,pmp_features_test,pmp_power_train,pmp_power_test, top_power_train,top_power_test);
    #Traning curve paper
    if iter==-1:
        mape_arr = [];
        for i in np.arange(0,1376720,1000):
            p_buf_lm = tree.DecisionTreeRegressor();
            p_buf_model = p_buf_lm.fit(features_concat_train[0:i+1,:],top_power_train[0:i+1]);
            p_buf_pred = p_buf_lm.predict(features_concat_test[0:i+1]);
            mape_decision_tree=return_mape(top_power_test[0:i+1],p_buf_pred);
            mape_arr.append(mape_decision_tree);
        """END"""
        print "Training curve:"
        print mape_arr;
    #print "Total single model:\n"
    #all_model(features_concat_train,top_power_train,1,features_concat_test,top_power_test);
    
    #only diff
    #all_model(features_concat_train,diff_power_train,1,features_concat_test,diff_power_test);
 
 

