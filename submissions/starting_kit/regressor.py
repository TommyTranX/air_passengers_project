from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg =xgb.XGBRegressor(colsample_bytree= 0.7, learning_rate= 0.05, max_depth= 5, min_child_weight=4, n_estimators= 1000, nthread= 4, objective='reg:linear', silent=1, subsample= 0.7)

        #self.enc=TargetEncoder(cols=['DateOfDeparture','Departure', 'Arrival', 'State_Dep','city_Dep','State_Arr','city_Arr'])




    def fit(self, X, y):
        #X = self.enc.fit_transform(X, y)
        self.reg.fit(X, y)
        

    def predict(self, X):
        return self.reg.predict(X)
