from numpy import *
from core.generalized import GeneralizedModel
from core.trainer import ActivationTrainer
from utils.helpers import initialize_weights

class MatrixFactor(GeneralizedModel):
    def __init__(self, shape, r, learn_rate=0.1, beta=0., momentum=0.):
        m, n = shape[0], shape[1]
        self.U = initialize_weights(m,r)
        self.G = initialize_weights(r,n)
        self.bias = zeros((m,n))
        self.shape = (m, n)
        self.r = r
        self.prevgrad = zeros(len(self.params))
        self.trainer = ActivationTrainer()
    
    @property
    def params(self):
        return hstack((self.U.flatten(), self.G.flatten(), self.bias().flatten))
    
    @params.setter
    def params(self, value):
        (m, n), r = self.shape, self.r
        start, end = 0, m*r
        self.U = reshape(value[start:end], (m,r))
        start, end = end, end + r*n
        self.G = reshape(value[start:end], (r,n))
        start, end = end, end + m*n
        self.bias = reshape(value[start:end], (m,n))
        
    def cost(self, data, activations):
        U, G = self.U, self.G
        sa = sum(activations)
        diff = (data - U.dot(G) - self.bias) * activations
        cost = .5 * sum(diff ** 2) / sa
        Ugrad = -diff.dot(G.T) / sa
        Ggrad = -U.T.dot(diff) / sa
        bgrad = -diff / sa
        if self.beta > 0:
            cost += .5 * self.beta * (sum(U**2) + sum(G**2))
            Ugrad += self.beta * U
            Ggrad += self.beta * G
        grad = hstack((Ugrad.flatten(), Ggrad.flatten(), bgrad.flatten()))
        return cost, grad

    def update(self, grad):
        self.params -= self.momentum * self.prevgrad + self.learn_rate * grad
        return self

    def train(self, data, activations, max_iter=1):
        args = { 'epochs': self.epochs,
                 'batch_size': self.batch_size,
                 'max_iter': max_iter,
                 'verbose': self.verbose }
        return self.trainer.train(self, data, activations, **args)
        
