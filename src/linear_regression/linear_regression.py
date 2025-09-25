import logging
from numpy.linalg import inv
import numpy as np

logger = logging.getLogger('linear_regression')

class LinearRegression:
    def __init__(self, use_intercept=True):
        self.coef = 0
        self.intercept = 0
        self.use_intercept = use_intercept
        
    def fit(self, X, y):
        R = inv(X.T @ X) @ X.T @ y
        self.coef = R[0]
        if self.use_intercept:
            self.intercept = R[1]
        logger.info(f"Model fitted: coef={self.coef}, intercept={self.intercept}")
    
    def predict(self, X):
        return X @ np.array([self.coef, self.intercept])
