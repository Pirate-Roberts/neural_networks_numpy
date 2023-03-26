import numpy as np
from sklearn.metrics import log_loss as log_loss_sklearn

class Perceptron():
    """
    Perceptron class that create  a perceptron
    model + all associated algorithms (training,
    inference, etc)
    """
    def __init__(self, input_n):
        self.n = input_n
        self.W = np.random.randn(self.n, 1)
        self.B = np.random.randn(1)
    
    def model(self, X):
        """
        Function that execute the perceptron model on data.
        Args:
            - X (array): input data
        """
        self.Z = np.dot(X, self.W) + self.B
        self.A = 1 / (1 + np.exp(-self.Z))

    def Log_Loss(self, y, eps=1e-15, loglosssklearn=True):
        """
        Perceptron's cost function.
        """
        if loglosssklearn:
            self.loss = log_loss_sklearn(y, self.A)
        else:
            self.loss = (
                -1/len(y) * np.sum(
                    y * np.log(self.A+eps) + 
                    (1-y) * np.log(1-self.A+eps)))
    
    def gradients(self, X, y):
        """
        
        """
        self.dW = 1/len(y) * np.dot(X.T, (self.A-y))
        self.dB = 1/len(y) * np.sum(self.A-y)
    
    def update(self, lr):
        """
        Update perceptron's coefficients
        """
        self.W = self.W - lr * self.dW
        self.B = self.B - lr * self.dB
        
    def fit(self, X, y, lr, cycle_nb, eps=1e-15, loglosssklearn=True):
        """
        Training function
        """
        training_loss = []
        for i in range(cycle_nb):
            self.model(X)
            self.Log_Loss(y, eps, loglosssklearn)
            self.gradients(X, y)
            self.update(lr)
            training_loss.append(self.loss)

        return training_loss

    def predict(self, X):
        """
        
        """
        self.model(X)
        pred_prob = self.A
        pred = pred_prob >= 0.5
        
        return pred_prob, pred
