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

    cat_cols = ['StoreType', 'Assortment', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'StateHoliday', 'month']

    for c in cat_cols:
        comb[c] = comb[c].astype('category')
#        train[c] = train[c].astype('category')
#        cats = train[c].cat.categories.tolist()
#        print(cats)
#        test[c] = pd.Categorical(values=test[c], categories = cats)

    comb = comb.drop(['CompNormDist', 'CompOpenDate', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Date', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'SchoolHoliday', 'StateHoliday' ], axis = 1)

    test = comb.loc[comb.Sales == -1,]
    train = comb.loc[comb.Sales != -1]
    return train, test;

def create_predictions_sales(train, test):
#    sub = train.loc[train.Store == 1]
#    sub = sub.drop(['Date', 'Store', 'CompOpen', 'StoreType', 'Assortment', 'CompDist'], axis = 1)
#    test = test.drop(['Date', 'Store', 'CompOpen', 'StoreType', 'Assortment', 'CompDist'], axis = 1)

    sub= train
    sub = sub.drop(['Date'], axis = 1)
    subtest = test.drop(['Date'], axis = 1)

    subtest.Open = subtest.Open.astype('int')
    sub = pd.get_dummies(sub)
    subtest = pd.get_dummies(subtest)

    target = np.array(sub.Sales)
    sub = sub.drop('Sales', axis = 1)
    cols = list(sub.columns)
    sub = np.array(sub)

    trn, tst, trgt_train, trgt_test = train_test_split(sub, target, test_size = .3, random_state = 42)

    def rmse(preds, target):
        error = np.sqrt(((preds-target) ** 2).mean())
        print(error)
        return(error)

    def mae(preds, target):
        error = np.mean(abs(preds-target))
        print(error)
        return(error)

#    rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
#    rf.fit(trn, trgt_train)
#    preds = rf.predict(tst)
#    rmse(preds, trgt_test)
#    mae(preds, trgt_test)
#
#    imprt = rf.feature_importances_
#    pd.DataFrame({
#            'var' : cols,
#            'importances' : imprt
#            })

    xg = XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0
                      , subsample=0.75,max_depth=7)
    xg.fit(X = trn, y = trgt_train)
    preds = xg.predict(tst)
    rmse(preds, trgt_test)
    mae(preds, trgt_test)

#    rid = BayesianRidge()
#    rid.fit(trn, trgt_train)
#    preds = rid.predict(tst)
#    rmse(preds, trgt_test)
#    mae(preds, trgt_test)
#
#    las = LassoLars(alpha = 1)
#    las.fit(trn, trgt_train)
#    preds = las.predict(tst)
#    rmse(preds, trgt_test)
#    mae(preds, trgt_test)
#
#    lin = LinearRegression()
#    lin.fit(trn, trgt_train)
#    preds = lin.predict(tst)
#    rmse(preds, trgt_test)
#    mae(preds, trgt_test)




def create_predictions_custs(train, test):
    #sub = train.loc[train.Store == 1]
    sub= train
    sub = sub.drop(['Sales'], axis = 1)
#    subtest = test.drop(['Date'], axis = 1)

    subtest.Open = subtest.Open.astype('int')
    sub = pd.get_dummies(sub)
    subtest = pd.get_dummies(subtest)

    target = np.array(sub.Customers)
    sub = sub.drop('Customers', axis = 1)
    cols = list(sub.columns)
    sub = np.array(sub)
    testcols = list(subtest.columns)
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

    xg = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
    xg.fit(X = trn, y = trgt_train)
    preds = xg.predict(tst)
    rmse(preds, trgt_test)
    mae(preds, trgt_test)

    joblib.dump(xg, "customers.joblib.dat")
    xg2 = joblib.load("customers.joblib.dat")

    preds2 = xg2.predict(tst)
    rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
    rf.fit(trn, trgt_train)
    preds = rf.predict(tst)
    rmse(preds, trgt_test)
    mae(preds, trgt_test)

    subtest = pd.DataFrame(subtest)
    subtest['Customers'] = preds

    return subtest

if __name__ == '__main__' :
    train, test = import_data()


#    test = create_predictions_custs(train, test)


