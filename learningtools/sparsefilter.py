import scipy as np

from numpy.random import uniform
from scipy import sum, sqrt, square, ones, zeros
from core.generalized import GeneralizedModel
from core.trainer import Trainer
from utils.helpers import initialize_weights
            
def compute_numerical_gradient(model, data, err=1e-8):
    params = model.params
    numgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)
    for i in range(len(params)):
        perturb[i] = err
        model.params = params - perturb
        loss1 = model.cost(data)[0]
        model.params = params + perturb
        loss2 = model.cost(data)[0]
        numgrad[i] = (loss2 - loss1)/(2*err)
        perturb[i] = 0
    model.params = params
    return numgrad

def l2row(X, eps=1e-8):
    N = sqrt(sum(square(X), 1) + eps)
    Y = (X.T / N).T
    return Y, N

def l2rowg(X,Y,N,D):
    g1 = D.T / N
    g2 = Y.T * sum(D*X, 1) / square(N)
    return (g1 - g2).T

class SparseFilter(GeneralizedModel):
    """
    Implements SparseFilter according to: 
    http://cs.stanford.edu/~jngiam/papers/NgiamKohChenBhaskarNg2011.pdf
    
    SparseFilter has been adapted to work with the learningtools toolbox.
    This includes support for stochastic gradient decent + momentum 
    
    TODO:
    Adapt SparseFilter to use arbitrary non-linear functions for the inital
    computation of f. This first requires a better understanding of the gradient
    computation.
    """
    attrs_ = ['size_in', 'size_out', 'learn_rate', 'epochs', 
              'batch_size', 'momentum', 'verbose']
              
    def __init__(self, size_in, size_out, learn_rate=0.1, epochs=1, 
                 batch_size=100, momentum=0.9, verbose=0):
        self.size_in = size_in
        self.size_out = size_out
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.momentum = momentum
        self.W = initialize_weights(size_in, size_out)
        self._prevgrad = zeros(self.W.flatten().shape)
        self.trainer = Trainer()
        
    @property
    def params(self):
        return self.W.flatten()
    
    @params.setter
    def params(self, value):
        self.W = np.reshape(value, (self.size_out, self.size_in))

    def propup(self, X, eps=1e-8):
        #~ F = self.W.dot(X.T) 
        F = X.dot(self.W.T).T
        Fs = sqrt(square(F) + eps)
        NFs, L2Fs = l2row(Fs)
        Fhat, L2Fn = l2row(NFs.T)
        return F, Fs, NFs, L2Fs, Fhat, L2Fn
    
    def backprop(self, X, F, Fs, NFs, L2Fs, Fhat, L2Fn):
        DeltaW = l2rowg(NFs.T, Fhat, L2Fn, ones(Fhat.shape))
        DeltaW = l2rowg(Fs, NFs, L2Fs, DeltaW.T)
        #~ DeltaW = (DeltaW * F / Fs).dot(X)
        DeltaW = X.T.dot((DeltaW * F / Fs).T).T
        return DeltaW
        
    def cost(self, X):
        # Feed Forward
        F, Fs, NFs, L2Fs, Fhat, L2Fn = self.propup(X)
        cost = sum(Fhat)
        
        # Backprop
        DeltaW = self.backprop(X, F, Fs, NFs, L2Fs, Fhat, L2Fn)
        grad = DeltaW.flatten()
        
        return cost, grad
        
    def update(self, grad):
        prevgrad = self._prevgrad
        dw = self.momentum * prevgrad + self.learn_rate * grad
        self.params -= dw
        self._prevgrad = dw
        return self

    def train(self, data, max_iter=1):
        args = { 'epochs': self.epochs,
                 'batch_size': self.batch_size,
                 'max_iter': max_iter,
                 'verbose': self.verbose }
        return self.trainer.train(self, data, **args)
