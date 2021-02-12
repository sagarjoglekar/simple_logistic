import numpy as np
from numpy import random


class LogisticRegression:
    def __init__(self,X_dim , regularization = None , learningRate = 0.0001 , lambda_ = 0.01): #regularization can be `L1` or `L2`
        self.b = 0.0
        self.w = np.zeros((X_dim,1))
        self.reg = regularization
        self._ll = lambda_
        self._lr = learningRate
        self.n = 1


    def sigmoid(self, X ):
        z = np.matmul(X, self.w) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self , yhat, y):
        cost = 0
        if self.reg == 'L2':
            cost = (-y * np.log(yhat) - (1 - y)*np.log(1 - yhat)).mean() +  self._ll*(np.matmul(self.w.T, self.w)) #Taking a mean across all samples
        else:
            cost = -(y * np.log(yhat) + (1 - y)*np.log(1 - yhat)).mean()  +  self._ll*(np.sum(self.w.T)) #Taking a mean across all samples
        return cost

    def predict(self, X):
        yhat = self.sigmoid(X)
        return yhat

    def loss_derivative_weight(self, X , yhat , y ):
        #yhat: Predicted Probabilities
        #y : class label
        #X :Features
        n = self.n
        dz = yhat - y
        regTerm = 0.0
        if self.reg:
            if self.reg == 'L1':
                regTerm = (self._ll/n)*np.max(self.w)
            else:
                regTerm = 2*(self._ll/n)*np.sum(self.w)
            
        dw = 1/n*(np.matmul(X.T , dz)) + regTerm
        return dw

    def loss_derivative_bias(self,  yhat , y):
        #yhat: Predicted Probabilities
        #y : class label
        dz = yhat - y        
        return (1/self.n)*np.sum(dz)


    def SGD(self , x_train, y_train , n_epoch, printFreq = 100):
        self.n = len(y_train)
        print(x_train.shape , y_train.shape)
        for epoch in range(n_epoch):
            loss = 0
            yhat = self.predict(x_train)
            loss += self.loss(yhat , y_train)
            dw = self.loss_derivative_weight(x_train,yhat,y_train)
            db = self.loss_derivative_bias(yhat, y_train)

            self.w = self.w - self._lr*dw
            self.b = self.b - self._lr*db
            if epoch % printFreq == 0:
                print("Logistic Loss for epoch %d is %f"%(epoch, loss.mean()))
    
    def _getModelParams(self):
        return {'w':self.w , 'b':self.b}
