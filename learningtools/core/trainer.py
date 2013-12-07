import time
import scipy as np
from numpy.random import permutation
from sys import stdout
from matplotlib.mlab import find
from scipy import sparse

def make_batches(data, labels=None, batch_size=100):
    if labels is not None:
        num_labels = labels.shape[1]
        cls_data = [data[find(labels[:,i] == 1)] for i in range(num_labels)]
        cls_sizes = [d.shape[0] for d in cls_data]
        cls_sels = [permutation(range(s)) for s in cls_sizes]
        n = min(cls_sizes) * len(cls_sizes)
        batch_size = min(n, batch_size)
        lpb = batch_size / num_labels
        new_dat = []
        for i in range(n/batch_size):
            for sel, cd in zip(cls_sels, cls_data):
                new_dat.append(cd[sel[i*lpb:(i+1)*lpb]])
        if sparse.issparse(data):
            data = sparse.vstack(new_dat).tocsr()
        else:
            data = np.vstack(new_dat)
        labels = np.tile(np.repeat(np.eye(num_labels),lpb,0), (n/batch_size,1))
        n = len(labels)
        perm = range(n)
    else:
        n = data.shape[0]
        perm = permutation(range(n))
    i = 0
    while i < n:
        batch = perm[i:i+batch_size]
        i += batch_size
        yield (data[batch], None) if labels is None else (data[batch], labels[batch])

#~ def make_batches(data, labels=None, batch_size=100):
    #~ n = data.shape[0]
    #~ perm = permutation(range(n))
    #~ i = 0
    #~ while i < n:
        #~ batch = perm[i:i+batch_size]
        #~ i += batch_size
        #~ yield (data[batch], None) if labels is None else (data[batch], labels[batch])
        
        
class Trainer(object):
    def __init__(self):
        self.cost_hist = []
        self.total_epochs = 0

    def __repr__(self):
        te = self.total_epochs
        ac = np.mean(self.cost_hist)
        nm = self.__class__.__name__
        return '%s\n    Total Epochs: %i\n    Average Cost: %.4f'%(nm, te, ac)
        
    def train(self, model, data, targets=None, epochs=1, batch_size=10, max_iter=1, verbose=0):
        #~ data = np.array(data)
        n, m = data.shape
        num_batches = n / batch_size
        e = 0
        
        if verbose: 
            start_time = time.clock()
        
        while e < epochs:
            e += 1
            self.total_epochs += 1
            batches = 0 
            for D, T in make_batches(data, targets, batch_size):
                batches += 1
                for i in range(max_iter):
                    cost, grad = model.cost(D) if T is None else model.cost(D, T)
                    model = model.update(grad)
                    self.cost_hist.append(cost)
                    if verbose >= 2:
                        print 'Batch %i - Iter %i - Cost %0.6f\r'%(batches, i+1, cost),
                        stdout.flush()
            if verbose >= 1:
                print 'Training Epoch %i'%(self.total_epochs), 
                print 'Average Cost: %0.6f\t\t'%np.mean(self.cost_hist[-num_batches*max_iter:])
                stdout.flush()

        if verbose >= 1: 
            end_time = time.clock()
            print 'Runtime %0.2fs'%(end_time-start_time)
            
        return model
