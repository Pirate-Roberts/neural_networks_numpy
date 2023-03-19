import numpy as np

def accuracy(y, y_pred):
    """
    
    """
    tp = TP(y, y_pred)
    tn = TN(y, y_pred)
    fp = FP(y, y_pred)
    fn = FN(y, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc


def precision(y, y_pred):
    """
    
    """
    tp = TP(y, y_pred)
    fp = FP(y, y_pred)
    prec = tp / (tp + fp)
    return prec


def recall(y, y_pred):
    """
    
    """
    tp = TP(y, y_pred)
    fn = FN(y, y_pred)
    rec = (tp) / (tp + fn)
    return rec


def TP(y, ypred):
    """
    
    """
    tp = np.dot(y.T, ypred)
    return tp


def TN(y, ypred):
    """
    
    """
    tn = np.dot((1 - y.T), (1 - ypred))
    return tn


def FP(y, ypred):
    """
    
    """
    fp = np.dot((1 - y.T), ypred)
    return fp


def FN(y, ypred):
    """
    
    """
    fn = np.dot(y.T, (1 - ypred))
    return fn