from numpy.linalg import inv
import numpy as np

class LinearRegression:
    def __init__(self, use_intercept=True):
        self.a=0
        self.b=0
        
    def fit(self, X, y):
        R=inv(X.T@X)@X.T@y
        self.a=R[0]
        self.b=R[1]
    
    def predict(self, X):
        return X@np.array([self.a,self.b])


X = np.array([[2, 4], [3, 5], [4, 6], [5, 7]])
y = np.array([10, 30, 40, 50])

clr = LinearRegression()
clr.fit(X,y)

print(clr.predict(X))

print(clr.predict(np.array([[1,20],[1,35],[1,60]])))