import pickle
import scipy as np

from trainer import Trainer

class GeneralizedModel(object):
    """
    Base class for learning models. All model types inheriting this object must
    include:
        : attrs_ class variable as an iterable of attributes to display for
          pretty printing (probably a better way to do this)
        : cost function that takes the model and data (and possibly labels)
          and computes the cost of the data given the model. Must return
          a numeric cost and a gradient
        : train function that takes the model and data (and possibly labels)
          and transforms the data into a suitable format to pass into an 
          optimization algorithm
        : update function that takes the model and gradient to update the model
          parameters
    
    This class adds support for pretty printing and saving using pythons
    pickle interface
    """
    attrs_ = []
    
    def __repr__(self):
        L = []
        for attr in self.attrs_:
            L.append(attr + '=' + str(getattr(self, attr)))
        return self.__class__.__name__ + '(' + ', '.join(L) + ')'

    def cost(self, data):
        raise Error('Method not implemented')
    
    def train(self, data):
        raise Error('Method not implemented')
    
    def update(self, grad):
        raise Error('Method not implemented')
        
    def save(self, filename, mode=2):
        f = open(filename, 'w')
        classname = self.__class__.__name__
        print 'Saving %s to %s under mode %i'%(self.__class__.__name__, filename, mode)
        pickle.dump(self, f, mode)
        f.close()
        

class GeneralizedBoltzmann(GeneralizedModel):
    attrs_ = ['trainfn', 'n', 'batch_size', 'epochs', 'learn_rate', 'beta', 'momentum', 'verbose']
    
    def __init__(self, trainfn='cdn', n=1, batch_size=10, epochs=1, learn_rate=0.1, 
                 beta=0.0001, momentum=0., verbose=False):
        self.trainfn = trainfn
        self.epochs = epochs
        self.n = n
        self.learn_rate = learn_rate
        self.beta = beta
        self.batch_size = batch_size
        self.momentum = momentum
        self.trainer = Trainer()
        self.verbose = verbose
        
    def gibbs_hvh(self, h, mf=False, **args):
        v_samples = self.propdown(h, **args)
        v = v_samples[0][1] if mf else v_samples[0][0]
        h_samples = self.propup(v, **args)
        return v_samples, h_samples
    
    def gibbs_vhv(self, v, mf=False, **args):
        h_samples = self.propup(v, **args)
        h = h_samples[-1][1] if mf else h_samples[-1][0]
        v_samples = self.propdown(h, **args)
        return v_samples, h_samples
    
    def cost(self, v):
        if len(np.shape(v)) == 1: v.shape = (1,len(v))
        use_fw = self.trainfn == 'fpcd'
        use_persist = use_fw or self.trainfn == 'pcd'
        num_points = v.shape[0]
        # positive phase
        pos_h_samples = self.propup(v)
        # negative phase
        nh0 = self.p[:num_points] if use_persist else pos_h_samples[-1][0]
        for i in range(self.n):
            neg_v_samples, neg_h_samples = self.gibbs_hvh(nh0, fw=use_fw)
            nh0 = neg_h_samples[-1][0]
        # compute gradients
        grads = self.grad(v, pos_h_samples, neg_v_samples, neg_h_samples)
        self.p[:num_points] = nh0
	# compute reconstruction error
        if self.trainfn=='cdn':
            cost = np.sum(np.square(v - neg_v_samples[0][1])) / self.batch_size
        else:
            cost = np.sum(np.square(v - self.gibbs_vhv(v)[0][0][1])) / self.batch_size
        return cost, grads
        
    def train(self, data, max_iter=1):
        args = { 'epochs': self.epochs,
                 'batch_size': self.batch_size,
                 'max_iter': max_iter,
                 'verbose': self.verbose }
        return self.trainer.train(self, data, **args)

