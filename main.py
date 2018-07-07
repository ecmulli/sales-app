# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 05:16:22 2018

@author: evanc
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import BayesianRidge, LassoLars, LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import preprocessing
import time

def import_data():

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    stores = pd.read_csv('store.csv')

    train = train.merge(stores, how = 'left', on = 'Store')
    test = test.merge(stores, how = 'left', on = 'Store')
    train.Date = train.Date.astype('datetime64[ns]')
    test.Date = test.Date.astype('datetime64[ns]')

    train['month'] = train.Date.map(lambda x: x.strftime('%m'))
    test['month'] = test.Date.map(lambda x: x.strftime('%m'))

    mini = train.CompetitionDistance.min()
    maxi = train.CompetitionDistance.max()
    train.loc[train.CompetitionDistance.isnull(), 'CompetitionDistance'] = maxi
    train['CompNormDist'] = (train.CompetitionDistance - mini)/(maxi-mini)
    train['CompDist'] = pd.cut(train.CompNormDist,3, labels = ['near', 'medium', 'far'])

    mini = test.CompetitionDistance.min()
    maxi = test.CompetitionDistance.max()
    test.loc[test.CompetitionDistance.isnull(), 'CompetitionDistance'] = maxi
    test['CompNormDist'] = (test.CompetitionDistance - mini)/(maxi-mini)
    test['CompDist'] = pd.cut(test.CompNormDist,3, labels = ['near', 'medium', 'far'])

    train.loc[train.CompetitionOpenSinceYear.isnull(), 'CompetitionOpenSinceYear'] = 2030
    train.loc[train.CompetitionOpenSinceMonth.isnull(), 'CompetitionOpenSinceMonth'] = 1
    test.loc[test.CompetitionOpenSinceYear.isnull(), 'CompetitionOpenSinceYear'] = 2030
    test.loc[test.CompetitionOpenSinceMonth.isnull(), 'CompetitionOpenSinceMonth'] = 1

    int_cols = ['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']
    for c in int_cols:
        print(c)
        train[c] = train[c].astype('int')
        test[c] = test[c].astype('int')

    train['CompOpenDate'] = train.CompetitionOpenSinceYear.map(str) + '-' + train.CompetitionOpenSinceMonth.map(str) + '-' + '01'
    test['CompOpenDate'] = test.CompetitionOpenSinceYear.map(str) + '-' + test.CompetitionOpenSinceMonth.map(str) + '-' + '01'
    train['CompOpen'] = train.CompOpenDate >= train.Date.astype(str)
    test['CompOpen'] = test.CompOpenDate >= test.Date.astype(str)

#    keep = ['Store', 'Sales', 'Customers', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StoreType', 'Assortment', 'month', 'CompDist', 'CompOpen']
#    train = train[keep]
#    keep = list(set(keep) - set(['Sales', 'Customers']))
#    test = test[keep]
    test = test.drop('Id', axis = 1)

    test['Customers'] = -1
    test['Sales'] = -1
    comb = train.append(test)

    comb = comb.drop(['CompNormDist', 'CompOpenDate', 'CompetitionDistance','Date','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'SchoolHoliday', 'StateHoliday' ], axis = 1)

    test = comb.loc[comb.Sales == -1,]
    train = comb.loc[comb.Sales != -1]
    return train, test;

def create_predictions_custs(train, test, load_or_run = 'run'):
    #sub = train.loc[train.Store == 1]
    sub= train
    sub = sub.drop(['Sales'], axis = 1)
    subtest = test.drop(['Sales', 'Customers'], axis = 1)

    subtest.Open = subtest.Open.astype('int')

    for c in sub.columns:
        if sub[c].dtype == 'object' or sub[c].dtype.name == 'category':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(sub[c].values))
            sub[c] = lbl.transform(sub[c].values)

    for c in subtest.columns:
        if subtest[c].dtypes == 'object' or subtest[c].dtype.name == 'category':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(subtest[c].values))
            subtest[c] = lbl.transform(subtest[c].values)

    target = np.array(sub.Customers)
    sub = sub.drop('Customers', axis = 1)
    sub = np.array(sub)
    subtest = np.array(subtest)

    trn, tst, trgt_train, trgt_test = train_test_split(sub, target, test_size = .3, random_state = 42)

    def rmse(preds, target):
        error = np.sqrt(((preds-target) ** 2).mean())
        print(error)
        return(error)

    def mae(preds, target):
        error = np.mean(abs(preds-target))
        print(error)
        return(error)

    param_grid = {
            'n_jobs':[4],
            'learning_rate': [.01,.1,.3],
            'max_depth': [10],
            'n_estimators':[300],
            'booster':['gbtree'],
            'gamma':[0],
            'subsample':[1],
            'colsample_bytree':[1]}
    start = time.time()
    xg = XGBRegressor(silent = 0,)
    xg = GridSearchCV(xg, param_grid)
    if load_or_run == 'load':
        xg = joblib.load("customers.joblib.dat")
        print('loaded')
    else:
        xg.fit(X = trn, y = trgt_train)
        print('ran')
    print(time.time() - start)
    predsgb = xg.predict(tst)
    rmse(predsgb, trgt_test)
    mae(predsgb, trgt_test)
    testpreds = xg.predict(subtest)


    joblib.dump(xg, "customers.joblib.dat")

    #originally tried: gamma-linear regression, random forest, lasso and bayesian ridge regression

    return testpreds;

def create_predictions_sales(train, test, load_or_run = 'run'):
    sub= train
    subtest = test.drop('Sales', axis = 1)

    subtest.Open = subtest.Open.astype('int')
    for c in sub.columns:
        if sub[c].dtype == 'object' or sub[c].dtype.name == 'category':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(sub[c].values))
            sub[c] = lbl.transform(sub[c].values)

    for c in subtest.columns:
        if subtest[c].dtypes == 'object' or subtest[c].dtype.name == 'category':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(subtest[c].values))
            subtest[c] = lbl.transform(subtest[c].values)

    target = np.array(sub.Sales)
    sub = sub.drop('Sales', axis = 1)
    sub = np.array(sub)
    subtest = np.array(subtest)

    trn, tst, trgt_train, trgt_test = train_test_split(sub, target, test_size = .3, random_state = 42)

    def rmse(preds, target):
        error = np.sqrt(((preds-target) ** 2).mean())
        print(error)
        return(error)

    def mae(preds, target):
        error = np.mean(abs(preds-target))
        print(error)
        return(error)

    param_grid = {
            'n_jobs':[4],
            'learning_rate': [.01,.1,.3],
            'max_depth': [10],
            'n_estimators':[500],
            'booster':['gbtree'],
            'gamma':[0],
            'subsample':[1],
            'colsample_bytree':[1]}
    start = time.time()
    xg = XGBRegressor(silent = 0)
    xg = GridSearchCV(xg, param_grid)
    if load_or_run == 'load':
        xg = joblib.load("sales.joblib.dat")
        print('loaded')

    else:
        xg.fit(X = trn, y = trgt_train)
        print('ran')

    print(xg.best_estimator_)
    print(time.time() - start)
    preds = xg.predict(tst)
    rmse(preds, trgt_test)
    mae(preds, trgt_test)
    custpreds = xg.predict(subtest)

    joblib.dump(xg, "sales.joblib.dat")

    return(custpreds);

if __name__ == '__main__' :
    train, test = import_data()
    test['Customers'] = create_predictions_custs(train,test, load_or_run = 'load')
    test['Sales'] = create_predictions_sales(train,test, load_or_run = 'load')
    test.to_csv('test_with_preds.csv')