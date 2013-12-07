from core.generalized import GeneralizedBoltzmann
import numpy as np

class DBN(GeneralizedBoltzmann):
    attrs_ = GeneralizedBoltzmann.attrs_ + ['num_layers']

    def __init__(self, layers, **args):
        GeneralizedBoltzmann.__init__(self, **args)
        self.layers = layers
        self.num_layers = len(self.layers)
        if self.persistent:
            self.p = np.zeros((self.batch_size, self.layers[-1].hidden_size))
            
    def update(self, grads, **args):
        for i,rbm in enumerate(self.layers):
            self.layers[i] = rbm.update([grads[i]])
        return self
        
    def params(self):
        return [r.params() for r in self.layers]
        
    def propup(self, v, stochastic=False):
        '''propagates v from the first layer to last layer
        returns the activations from each layer in a list'''
        activations = []
        for net in self.layers:
            h, h_real, h_pre_non_lin = net.propup(v)[0]
            activations.append((h, h_real, h_pre_non_lin))
            v = h if stochastic else h_real
        return activations
    
    def propdown(self, h, stochastic=False):
        '''propagates h from the last layer to the first layer
        returns the activations from each layer in a list'''
        activations = []
        for net in reversed(self.layers):
            v, v_real, v_pre_non_lin = net.propdown(h)[0]
            activations.insert(0, (v, v_real, v_pre_non_lin))
            h = v if stochastic else v_real
        return activations
    
    def E(self, v0, h_chain):
        '''calculates the total energy of the DBN with visible state v and
        the hidden state of each layer given in h_chain'''
        energy = 0
        for (i,net) in enumerate(self.layers):
            energy += net.E(v0, [h_chain[i]])
            v0 = h_chain[i]
        return energy

    def greedytrain(self, data):
        X = data
        for rbm in self.layers:
            rbm = rbm.train(X)
            X = rbm.propup(X)[0][1]
        return self
