import numpy as np

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

    def Log_Loss(self, y):
        """
        Perceptron's cost function.
        """
        self.loss = -1/len(y) * np.sum(y * np.log(self.A) + (1-y) * np.log(1-self.A))
    
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
        
    def fit(self, X, y, lr, cycle_nb):
        """
        Training function
        """
        training_loss = []
        for i in range(cycle_nb):
            self.model(X)
            self.Log_Loss(y)
            self.gradients(X, y)
            self.update(lr)
            training_loss.append(self.loss)

        return training_loss


