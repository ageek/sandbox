from core.generalized import GeneralizedBoltzmann
from utils.functions import *
from utils.helpers import initialize_weights

from scipy import *
from numpy.random import normal, permutation, rand, uniform

LAYER_MODEL_FNS = { 'binary': sigmoid, 
                    'linear': linear,
                    'noisylinear': linear, 
                    'NRLU': rectified_linear,
                    'softmax': softmax,
                    'poisson': sigmoid }

LAYER_SAMPLE_FNS = { 'binary': sample_bernoulli, 
                     'linear': linear,
                     'noisylinear': sample_gaussian,
                     #~ 'NRLU': rectified_linear,
                     'NRLU': sample_NRLU,
                     'softmax': sample_softmax,
                     'poisson': sample_poisson }

class RBM(GeneralizedBoltzmann):
    attrs_ = GeneralizedBoltzmann.attrs_ + ['hidden_size', 'visible_size', 
             'hidden_layer', 'visible_layer', 'dropout']
    
    def __init__(self, visible_size=25, hidden_size=50, visible_layer='binary', 
                 hidden_layer='binary', dropout=0.0, **args):
        # Initialize args
        GeneralizedBoltzmann.__init__(self, **args)
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        # Initialize Biases and Weights
        self.vbias = zeros(visible_size)
        self.hbias = zeros(hidden_size)
        self.W = initialize_weights(visible_size, hidden_size)
        self.prevgrad = {'W': zeros(self.W.shape), 
                         'hbias': zeros(hidden_size), 
                         'vbias': zeros(visible_size)}
        if self.trainfn == 'fpcd' or self.trainfn == 'pcd':
            self.p = np.zeros((self.batch_size, self.hidden_size))
        if self.trainfn == 'fpcd':
            self.fW = zeros(self.W.shape)
            self.flr = self.learn_rate*exp(1) #fast learn rate heuristic
        
    def params(self):
        return {'W': self.W, 'hbias': self.hbias, 'vbias': self.vbias}
    
    def propup(self, vis, fw=False):
        f = LAYER_MODEL_FNS[self.hidden_layer]
        g = LAYER_SAMPLE_FNS[self.hidden_layer]
        W = self.fW + self.W if fw else self.W
        pre_non_lin = dot(W, vis.T).T + self.hbias
        non_lin = f(pre_non_lin)
        if self.dropout > 0.0:
            activs = uniform(0, 1, size=non_lin.shape) >= self.dropout
            non_lin *= activs
        sample = g(non_lin) if self.hidden_layer != 'NRLU' else g(pre_non_lin * activs)
        return [(sample, non_lin, pre_non_lin)]
    
    def propdown(self, hid, fw=False):
        f = LAYER_MODEL_FNS[self.visible_layer]
        g = LAYER_SAMPLE_FNS[self.visible_layer]
        W = self.fW + self.W if fw else self.W
        pre_non_lin = dot(W.T, hid.T).T + self.vbias
        non_lin = f(pre_non_lin)
        sample = g(non_lin)
        return [(sample, non_lin, pre_non_lin)]
        
    def update(self, grads):
        grad = grads[0]
        prev_grad = self.prevgrad
        dW = self.momentum * prev_grad['W'] + self.learn_rate * (grad['W'] - self.beta * self.W)
        dh = self.momentum * prev_grad['hbias'] + self.learn_rate * grad['hbias']
        dv = self.momentum * prev_grad['vbias'] + self.learn_rate * grad['vbias']
        self.W += dW
        self.hbias += dh
        self.vbias += dv
        # Fast weight update for PCD
        if self.trainfn == 'fpcd':
            self.fW = (49./50)*self.fW + self.flr * grad['W'] 
        self.prevgrad['W'] = dW
        self.prevgrad['hbias'] = dh
        self.prevgrad['vbias'] = dv
        return self
    
    def grad(self, pv0, pos_h, neg_v, neg_h):
        grad = {}
        num_points = pv0.shape[0]
        E_v = neg_v[0][1]
        E_h = neg_h[0][1]
        E_hgv = pos_h[0][1]
        # Experimental 
        # Based on my calculations, the MLE gradient for
        # d/dW (log p(v, h)) =
        #       h * v.T - d/dW log p(h|v;W) - (<h * v.T> - d/dW log p(h|v;W))
        # This is because we do not know the state h associated with v
        # but assume it is from a model p(h|v) which we can differentiate 
        # 
        # In this experimental code, I compute the gradient term of
        # d/dW log p(h|v;W) for a bernoulli hidden unit on the up and down pass
        #~ E_vh = np.dot(2 * E_h.T - 1, E_v)
        #~ E_vhgv = np.dot(2 * E_hgv.T - 1, pv0)
        # Traditional gradient
        E_vh = np.dot(E_h.T, E_v)
        E_vhgv = np.dot(E_hgv.T, pv0)
        grad['W'] = (E_vhgv - E_vh) / num_points
        grad['vbias'] = mean(pv0 - E_v, 0)
        grad['hbias'] = mean(E_hgv - E_h, 0)
        return [grad]
    
    def E(self, v0, h0):
        if len(shape(v0)) == 1: v0.shape = (1,len(v0))
        if len(shape(h0[0])) == 1: h0.shape = (1,len(h0[0]))
        if self.visible_layer == 'linear':
            vis_e = sum(square(self.vbias - v0))/2
        else:
            vis_e = -sum(self.vbias * v0)
        if self.hidden_layer == 'linear':
            hid_e = sum(square(self.hbias - h0))/2
        else:
            hid_e = -sum(self.hbias * h0)
        vishid_e = -sum(dot(h0[0].T, v0) * self.W)
        return hid_e + vishid_e

    def F(self, v0):
        if len(shape(v0)) == 1: v0.shape = (1,len(v0))
        X = dot(v0, self.W.T) + self.hbias
        return -dot(v0, self.vbias) - sum(log(1 + exp(X)))

class DiscriminativeRBM(RBM):
    attrs_ = RBM.attrs_ + ['num_labels']
    
    def __init__(self, num_labels=10, **args):
        RBM.__init__(self, **args)
        self.num_labels = num_labels
        self.D = uniform(-1, 1, size=(self.hidden_size, num_labels)) / sqrt(self.hidden_size + num_labels)
        self.dbias = zeros(num_labels)
        self.prevgrad.update({'D': zeros(self.D.shape), 
                              'dbias': zeros(num_labels)})
        
    def propup(self, vis, fw=False):
        if len(shape(vis)) == 1: vis.shape = (1,len(vis))
        targets, data = vis[:,:self.num_labels], vis[:,self.num_labels:]
        f = LAYER_MODEL_FNS[self.hidden_layer]
        g = LAYER_SAMPLE_FNS[self.hidden_layer]
        W = self.fW + self.W if fw else self.W
        pre_non_lin = dot(data, W.T) + self.hbias + dot(targets, self.D.T)
        non_lin = f(pre_non_lin)
        sample = g(non_lin)
        return [(sample, non_lin, pre_non_lin)]
    
    def propdown(self, hid, fw=False):
        f = LAYER_MODEL_FNS[self.visible_layer]
        g = LAYER_SAMPLE_FNS[self.visible_layer]
        W = self.fW + self.W if fw else self.W
        targets, data = dot(hid, self.D), dot(hid, W) + self.vbias
        non_lin = (sigmoid(targets), f(data))
        sample = (sample_bernoulli(non_lin[0]), g(non_lin[1]))
        return [(hstack(sample), hstack(non_lin), hstack((targets, data)))]

    def grad(self, pv0, pos_h, neg_v, neg_h):
        grad = {}
        num_points = pv0.shape[0]
        targets, data = pv0[:,:self.num_labels], pv0[:,self.num_labels:]
        E_t = neg_v[0][1][:,:self.num_labels]
        E_v = neg_v[0][1][:,self.num_labels:]
        E_h = neg_h[0][1]
        E_hgv = pos_h[0][1]
        E_vh = np.dot(E_h.T, E_v)
        E_vhgv = np.dot(E_hgv.T, data)
        E_th = np.dot(E_h.T, E_t)
        E_thgv = np.dot(E_hgv.T, targets)
        grad['W'] = (E_vhgv - E_vh) / num_points
        grad['D'] = (E_thgv - E_th) / num_points
        grad['dbias'] = mean(targets - E_t)
        grad['vbias'] = mean(data - E_v, 0)
        grad['hbias'] = mean(E_hgv - E_h, 0)
        return [grad]

    def update(self, grads):
        RBM.update(self, grads)
        grad = grads[0]
        prev_grad = self.prevgrad
        dD = self.momentum * prev_grad['D'] + self.learn_rate * (grad['D'] - self.beta * self.D)
        dd = self.momentum * prev_grad['dbias'] + self.learn_rate * grad['dbias']
        self.D += dD
        self.dbias += dd
        self.prevgrad['D'] = dD
        self.prevgrad['dbias'] = dd
        return self
    
    def train(self, data, targets):
        y_label = zeros((len(targets), self.num_labels))
        for i, t in enumerate(targets):
            y_label[i, t] = 1
        return RBM.train(self, hstack([y_label, data]))
    
    def predict(self, data, mf=True):
        num_pts = 1 if len(data.shape) == 1 else data.shape[0]
        if mf:
            y_field = 0.5 * ones((num_pts, self.num_labels))
            vhv = self.gibbs_vhv(hstack((y_field, data)), mf=True)
            pred = argmax(vhv[0][0][1][:,:self.num_labels], 1)
        else:
            xup = self.propup(data)
        return pred
