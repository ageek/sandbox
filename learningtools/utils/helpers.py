from numpy.random import uniform
import numpy as np

def initialize_weights(sz_in, sz_out):
    return (uniform(-1, 1, size=(sz_out, sz_in)) / np.sqrt(sz_in + sz_out))

def compute_numerical_gradient(model, data, targets=None, err=1e-8):
    params = model.params
    numgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)
    if targets is None:
        costfn = lambda: model.cost(data)
    else:
        costfn = lambda: model.cost(data, targets)
    for i in range(len(params)):
        perturb[i] = err
        model.params = params - perturb
        loss1 = costfn()[0]
        model.params = params + perturb
        loss2 = costfn()[0]
        numgrad[i] = (loss2 - loss1)/(2*err)
        perturb[i] = 0
    model.params = params
    return numgrad
