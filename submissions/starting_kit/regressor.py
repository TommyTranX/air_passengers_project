from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class Regressor(BaseEstimator):
    def __init__(self):
        '''GB = GradientBoostingRegressor(n_estimators=5000,
            learning_rate=0.05, max_depth=3, max_features='sqrt',
            min_samples_leaf=15, min_samples_split=10, loss='huber')'''
        self.reg = xgb.XGBRegressor(colsample_bytree= 0.7, learning_rate= 0.05, max_depth= 7, min_child_weight=4, n_estimators= 15000, nthread= 4, objective='reg:linear', silent=1, subsample= 0.7)
        #self.reg =  AveragingModels(models = (GB, xgb_model))


        #self.enc=TargetEncoder(cols=['DateOfDeparture','Departure', 'Arrival', 'State_Dep','city_Dep','State_Arr','city_Arr'])




    def fit(self, X, y):
        #X = self.enc.fit_transform(X, y)
        self.reg.fit(X, y)
        

    def predict(self, X):
        return self.reg.predict(X)
