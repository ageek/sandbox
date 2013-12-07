from core.generalized import GeneralizedModel
from utils.helpers import initialize_weights

import utils.functions as fns 

import scipy as np
from scipy import sparse
from numpy.random import uniform

MODELFNS = { 'sigmoid': fns.sigmoid,
             'softmax': fns.softmax,
             'tanh': np.tanh,
             'linear': fns.linear,
             'ReLU': fns.rectified_linear }
GRADFNS = { 'sigmoid': fns.sigmoid_grad,
            'softmax': fns.softmax_grad,
            'tanh': fns.tanh_grad,
            'linear': fns.linear_grad,
            'ReLU': fns.rectified_linear_grad }

class MLPLayer(GeneralizedModel):
    attrs_ = ['size_in', 'size_out', 'modelfn', 'dropout']
    
    def __init__(self, size_in=10, size_out=10, modelfn='sigmoid', dropout=0.0):
        self._W = initialize_weights(size_in+1, size_out).astype(np.float128)
        self._size_in = size_in
        self._size_out = size_out
        self._modelfn = modelfn
        self.dropout = dropout
        
    @property
    def size_in(self):
        return self._size_in
    
    @property
    def size_out(self):
        return self._size_out

    @property
    def modelfn(self):
        return self._modelfn

    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        self._W = value
    
    @property
    def l2_penalty(self):
        tmp = np.zeros(self.W.shape)
        tmp[:,1:] = self.W[:,1:]
        return tmp.flatten()
        
    def propup(self, X, ispred=False):
        num_pts = np.shape(X)[0]
        X0 = np.hstack([np.ones((num_pts, 1)), X])
        f = MODELFNS[self.modelfn]
        W = self.W
        if ispred:
            W = np.copy(W)
            W[:,1:] *= (1-self.dropout)
        pre_non_lin = X0.dot(W.T)
        non_lin = f(pre_non_lin)
        if self.dropout > 0.0 and not ispred:
            non_lin *= uniform(0, 1, size=non_lin.shape) >= self.dropout
        return non_lin, pre_non_lin

    def backprop(self, A_in, Z_out, prev_delta, prev_params):
        f = GRADFNS[self.modelfn]
        num_pts = np.shape(Z_out)[0]
        bias_ones = np.ones((num_pts, 1))
        sgrd = f(np.hstack([bias_ones, Z_out]))
        delta = np.dot(prev_params.T, prev_delta) * sgrd.T
        grad = np.dot(delta[1:,:], np.hstack([bias_ones, A_in])) / num_pts
        return grad, delta

class SparseMLPLayer(MLPLayer):
    """
    MLP Layer that assumes input into the layer is given as a scipy.sparse
    data type
    """
    def propup(self, X, ispred=False):
        num_pts = np.shape(X)[0]
        X0 = sparse.hstack([np.ones((num_pts, 1)), X]).tocsr()
        f = MODELFNS[self.modelfn]
        W = self.W
        if ispred:
            W[:,1:] /= (1-self.dropout)
        pre_non_lin = X0.dot(W.T)
        non_lin = f(pre_non_lin)
        if self.dropout > 0.0 and not ispred:
            non_lin *= uniform(0, 1, size=non_lin.shape) >= self.dropout
        return non_lin, pre_non_lin

    def backprop(self, A_in, Z_out, prev_delta, prev_params):
        f = GRADFNS[self.modelfn]
        num_pts = np.shape(Z_out)[0]
        bias_ones = np.ones((num_pts, 1))
        sgrd = f(np.hstack([bias_ones, Z_out]))
        delta = np.dot(prev_params.T, prev_delta) * sgrd.T
        A = sparse.hstack([bias_ones, A_in]).tocsr()
        grad = A.T.dot(delta[1:,:].T).T / num_pts
        return grad, delta
    
class TopLayer(MLPLayer):
    attrs_ = MLPLayer.attrs_ + ['errorfn']
    
    def __init__(self, errorfn='sqrerr', **args):
        MLPLayer.__init__(self, **args)
        self._errorfn = errorfn
        
    @property
    def errorfn(self):
        return self._errorfn
        
    def backprop(self, A_in, Z_out, prediction, targets):
        num_pts = np.shape(A_in)[0]
        bias_ones = np.ones((num_pts, 1))
        delta = (prediction - targets).T
        if self.errorfn == 'sqrerr': 
            delta *= GRADFNS[self.modelfn](Z_out.T)
        grad = np.dot(delta, np.hstack([bias_ones, A_in])) / num_pts
        return grad, delta
