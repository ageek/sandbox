from core.generalized import GeneralizedModel
from core.trainer import Trainer
import utils.functions as fns

import scipy as np
from numpy.random import uniform

ERRORFNS = { 'logerr': fns.cross_entropy,
             'logerrsm': fns.cross_entropy_softmax,
             'sqrerr': fns.square_error }
             
class MLP(GeneralizedModel):
    attrs_ = ['num_layers', 'dims', 'learn_rate', 'beta', 'epochs', 'lr_decay',
              'batch_size', 'momentum', 'dropout', 'verbose']
    
    def __init__(self, layers=[], learn_rate=0.1, beta=0., epochs=1, momentum=0.,
                  batch_size=10, verbose=False, dropout=0.0, lr_decay=0.):
        self._layers = layers
        self.learn_rate = learn_rate
        self.beta = beta 
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dropout = dropout
        self.lr_decay = lr_decay
        self._num_layers = len(layers)
        self._dims = [layer.size_in for layer in layers] + [layers[-1].size_out]
        self._prevgrad = np.zeros(len(self.params))
        self.trainer = Trainer()
    
    @property
    def dims(self):
        return self._dims
    
    @property
    def num_layers(self):
        return self._num_layers
    
    @property
    def params(self):
        params = [layer.W.flatten() for layer in self._layers]
        return np.hstack(params)
        
    @params.setter
    def params(self, value):
        pos = 0
        for layer in self._layers:
            end = pos + (layer.size_in+1) * layer.size_out
            layer.W = np.reshape(value[pos:end], (layer.size_out, layer.size_in + 1))
            pos = end
    
    def update(self, grad):
        prevgrad = self._prevgrad
        tot_epoch = self.trainer.total_epochs
        learn_rate = 1. / (1 + tot_epoch * self.lr_decay) * self.learn_rate
        # Compute L2 norm gradient
        l2_norms = []
        for layer in self._layers:
            l2_norms.append(layer.l2_penalty)
        l2_norms = np.hstack(l2_norms)
        new_grad = grad + self.beta * l2_norms
        dw = self.momentum * prevgrad + learn_rate * new_grad
        self.params -= dw
        self._prevgrad = dw
        return self
        
    def propup(self, X, ispred=False):
        A = X
        if self.dropout > 0.0 and not ispred:
            A *= uniform(0, 1, size=A.shape) >= self.dropout
        results = [(A,None)]
        for layer in self._layers:
            results.append(layer.propup(A, ispred))
            A = results[-1][0]
        return results
    
    def backprop(self, propup_results, targets):
        results = []
        for i in range(self.num_layers, 0, -1):
            A_in = propup_results[i-1][0]
            Z_out = propup_results[i][1]
            if i == self.num_layers:
                prediction = propup_results[i][0]
                grad, delta = self._layers[i-1].backprop(A_in, Z_out, prediction, targets)
            else:
                grad, delta = self._layers[i-1].backprop(A_in, Z_out, delta, self._layers[i].W)
                delta = delta[1:,:]
            results.insert(0, (grad, delta))
        return results 

    def cost(self, data, targets):
        num_pts = data.shape[0]
        propup_results = self.propup(data)
        backprop_results = self.backprop(propup_results, targets)
        f = ERRORFNS[self._layers[-1].errorfn]
        pred = propup_results[-1][0]
        cost = f(pred, targets) / num_pts
        grad = np.hstack([grad.flatten() for grad, delta in backprop_results])
        return cost, grad
    
    def train(self, data, targets, max_iter=1):
        if self._layers[-1].modelfn in {'sigmoid', 'softmax'}:
            neglabel = 0
            poslabel = 1
        elif self._layers[-1].modelfn == 'tanh':
            neglabel = -1
            poslabel = 1
        y_label = neglabel * np.ones((len(targets), self.dims[-1]))
        for i, t in enumerate(targets):
            y_label[i, t] = poslabel
        args = { 'epochs': self.epochs,
                 'batch_size': self.batch_size,
                 'max_iter': max_iter,
                 'verbose': self.verbose }
        return self.trainer.train(self, data, y_label, **args)

    def predict(self, data):
        propup_results = self.propup(data, ispred=True)
        probs = propup_results[-1][0]
        return np.argmax(probs, 1)

class RMSPROP:
    """
    RMSPROP is an experimental update function for updating MLP parameters
    
    The idea is to ignore the magnitude of the gradient and only use
    the direction. Then an individual learn rate is learned for each
    parameter
    
    In order to work gracefully with mini-batches, the gradient is
    divided by the mean squared value of the gradient which is smoothed
    across mini-batches
    
    Note: This is still very buggy!
    """
    def __init__(self, net, inc=1.3, dec=0.5, ms_dec=0.9):
        self.inc = inc
        self.dec = dec
        self.ms = 1e4
        self.ms_dec = ms_dec
        self.factor = None
        self.net = net
        
    def __call__(self, grad):
        net = self.net
        if self.factor is None: self.factor = np.ones(net.params.shape)
        prevgrad = net._prevgrad
        momentum = net.momentum
        # Compute L2 norm gradient
        l2_norms = []
        for layer in net._layers:
            l2_norms.append(layer.l2_penalty)
        l2_norms = np.hstack(l2_norms)
        new_grad = grad + net.beta * l2_norms
        self.ms = self.ms_dec * self.ms + (1 - self.ms_dec) * np.dot(new_grad, new_grad)
        prevgrad_dir = (prevgrad > 0).astype('b') - (prevgrad < 0).astype('b')
        currgrad_dir = (new_grad > 0).astype('b') - (new_grad < 0).astype('b')
        self.factor *= self.inc * (prevgrad_dir == currgrad_dir) + \
                       self.dec * (prevgrad_dir != currgrad_dir)
        new_grad = currgrad_dir * self.factor
        dw = momentum * prevgrad + net.learn_rate * new_grad / np.sqrt(self.ms)
        net.params -= dw
        net._prevgrad = dw
        return net
