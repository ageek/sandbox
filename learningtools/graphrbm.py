from core.generalized import GeneralizedModel
from utils.functions import *

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
                     'NRLU': rectified_linear,
                     #~ 'NRLU': sample_NRLU,
                     'softmax': sample_softmax,
                     'poisson': sample_poisson }

def init_bias(opts):
    biases = []
    for o in opts:
        biases.append(zeros(o['size']))
    return biases

def init_weights(vopts, hopts, edges, randomize=True):
    weights = {}
    for i, j in edges:
        vsize, hsize = vopts[i]['size'], hopts[j]['size']
        if randomize:
            weights[i,j] = uniform(-1, 1, size=(hsize, vsize)) / sqrt(hsize + vsize)
        else:
            weights[i,j] = zeros((hsize, vsize))
    return weights

def init_persist(h_opts, batch_size):
    persist = []
    for o in opts:
        persist.append(zeros((batch_size, o['size'])))
    return persist
    
class GraphRBM(GeneralizedModel):
    attrs_ = ['trainfn', 'n', 'batch_size', 'epochs', 'learn_rate', 
              'beta', 'momentum', 'verbose']
    def __init__(self, vis_opts, hid_opts, edges, trainfn='fpcd', n=1, 
                 batch_size=100, epochs=1, learn_rate = 0.1, beta=0., 
                 momentum=0., verbose=0):
        self.vis_opts = vis_opts
        self.hid_opts = hid_opts
        self.edges = edges
        self.vbiases = init_bias(vis_opts)
        self.hbiases = init_bias(hid_opts)
        self.weights = init_weights(vis_opts, hid_opts, edges)
        self.fast_weights = init_weights(vis_opts, hid_opts, edges, False)
        self.persist_chain = init_persist(hid_opts, batch_size)
        self.trainfn = trainfn
        self.n = n
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.flr = self.learn_rate*exp(1)
        self.beta = beta
        self.momentum = momentum
        self.verbose = verbose
        self.prevgrad = None
    
    def propup(self, vis, fw=False):
        hid_act = [h for h in self.hbiases] 
        for i,j in self.edges:
            W = self.fast_weights[i,j] + self.weights[i,j] if fw else self.weights[i,j]
            hid_act[j] = hid_act[j] + dot(vis[i], W.T)
        for h_opt in self.hid_opts:
            f = LAYER_MODEL_FNS[h_opt['type']]
            g = LAYER_SAMPLE_FNS[h_opt['type']]
            non_lin = [f(h) for h in hid_act]
            sample = [g(h) for h in non_lin]
        return sample, non_lin, hid_act
    
    def propdown(self, hid, fw=False):
        vis_act = [v for v in self.vbiases]
        for i,j in self.edges:
            W = self.fast_weights[i,j] + self.weights[i,j] if fw else self.weights[i,j]
            vis_act[i] = vis_act[i] + dot(hid[j], W)
        for v_opt in self.vis_opts:
            f = LAYER_MODEL_FNS[v_opt['type']]
            g = LAYER_SAMPLE_FNS[v_opt['type']]
            non_lin = [f(v) for v in vis_act]
            sample = [g(v) for v in non_lin]
        return sample, non_lin, hid_act
    
    def gibbs_hvh(self, h, mf=False, **args):
        v_samples = self.propdown(h, **args)
        v = v_samples[1] if mf else v_samples[0]
        h_samples = self.propup(v, **args)
        return v_samples, h_samples
    
    def gibbs_vhv(self, v, mf=False, **args):
        h_samples = self.propup(v, **args)
        h = h_samples[1] if mf else h_samples[0]
        v_samples = self.propdown(h, **args)
        return v_samples, h_samples

    def cost(self, v):
        use_fw = self.trainfn == 'fpcd'
        use_persist = use_fw or self.trainfn == 'pcd'
        num_points = v[0].shape[0]
        # positive phase
        pos_h_samples = self.propup(v)
        # negative phase
        nh0 = self.persist_chain if use_persist else pos_h_samples[0]
        for i in range(self.n):
            neg_v_samples, neg_h_samples = self.gibbs_hvh(nh0, fw=use_fw)
            nh0 = neg_h_samples[0]
        # compute gradients
        grads = self.grad(v, pos_h_samples, neg_v_samples, neg_h_samples)
        self.persist_chain = nh0
        # compute reconstruction error
        if self.trainfn=='cdn':
            cost = sum([sum(square(v[i] - neg_v_samples[1][i])) / self.batch_size for i, vis in enumerate(v)])
        else:
            cost = sum([sum(square(v[i] - self.gibbs_vhv(v)[0][1][i])) / self.batch_size for i, vis in enumerate(v)])
        return cost, grads

    def grad(self, pv0, pos_h, neg_v, neg_h):
        grad = {'W':{},'hbias':{},'vbias':{}}
        num_points = pv0[0].shape[0]
        E_v = neg_v[1]
        E_h = neg_h[1]
        E_hgv = pos_h[1]
        for i,j in self.edges:
            E_vh = np.dot(E_h[j].T, E_v[i])
            E_vhgv = np.dot(E_hgv[j].T, pv0[i])
            grad['W'][i,j] = (E_vhgv - E_vh) / num_points
        for i, v in enumerate(pv0)
            grad['vbias'][i] = mean(pv0[i] - E_v[i], 0)
        for j, h in enumerate(pos_h):
            grad['hbias'][j] = mean(E_hgv[j] - E_h[j], 0)
        return grad

    def update(self, grad):
        dW = {}
        dh = {}
        dv = {}
        for i,j in self.edges:
            dW[i,j] = self.momentum * prev_grad['W'][i,j] + self.learn_rate * (grad['W'][i,j] - self.beta * self.weights[i,j])
            self.weights[i,j] += dW[i,j]
            # Fast weight update for FPCD
            if self.trainfn == 'fpcd':
                self.fast_weights[i,j] = (49./50)*self.fast_weights[i,j] + self.flr * grad['W'][i,j]
        for i, v in enumerate(self.vbiases):
            dv[i] = self.momentum * prev_grad['vbias'][i] + self.learn_rate * grad['vbias'][i]
            self.vbiases[i] += dv[i]
        for j, h in enumerate(self.hbiases):
            dh[j] = self.momentum * prev_grad['hbias'][j] + self.learn_rate * grad['hbias'][j]
            self.hbiases[j] += dh[j]
        # Fast weight update for P
        self.prevgrad['W'] = dW
        self.prevgrad['hbias'] = dh
        self.prevgrad['vbias'] = dv
        return self
