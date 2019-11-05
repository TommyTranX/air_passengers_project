from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder
from sklearn.neural_network import MLPRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = MLPRegressor()
        self.targ_enc = TargetEncoder(cols=['Departure','Arrival'], smoothing=8, min_samples_leaf=5)

        

    def fit(self, X, y):
        self.targ_enc.fit(X, y)
        X_train_te = self.targ_enc.transform(X.reset_index(drop=True))
        self.reg.fit(X_train_te.values, y)
        
    def predict(self, X):
        X_train_te = self.targ_enc.transform(X.reset_index(drop=True))
        return self.reg.predict(X_train_te.values)