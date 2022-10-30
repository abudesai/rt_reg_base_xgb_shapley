#Import required libraries
from random import Random
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from xgboost import XGBRegressor


model_fname = "model.save"
MODEL_NAME = "reg_base_xgb_shapley"

class Regressor(): 
    
    def __init__(self, n_estimators=250, eta=0.3, gamma=0.0, max_depth=5, **kwargs) -> None:
        self.n_estimators = n_estimators
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.model = self.build_model(**kwargs)     
        
        
    def build_model(self, **kwargs): 
        model = XGBRegressor(
            n_estimators=self.n_estimators,
            eta=self.eta,
            gamma=self.gamma,
            max_depth=self.max_depth,
            verbosity = 0, **kwargs, )
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit( X = train_X, y = train_y)
    
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        regressor = joblib.load(os.path.join(model_path, model_fname))
        return regressor


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    model = Regressor.load(model_path) 
    return model


def get_data_based_model_params(data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    return { }


