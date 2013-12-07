from scipy import exp, sum, sqrt, pi
from slice_sampler import slice_sampler 
from matplotlib.mlab import isvector 
import scipy as np

def sigmoid(X):
    """
    numpy.array -> numpy.array
    
    Compute sigmoid function: 1 / (1 + exp(-X))
    """
    return 1 / (1 + exp(-X))

def poisson1(L, t=1):
    return L * exp(-L*t)

def sample_poisson(L, t=1):
    return sample_bernoulli(poisson1(L, t))
    
def sigmoid_grad(X):
    sig = sigmoid(X)
    return sig * (1-sig)

def softmax_grad(X):
    sm = softmax(X)
    return sm*(1-sm)
    
def tanh_grad(X):
    return 1 / np.square(np.cosh(X))

def linear_grad(X):
    return np.ones(X.shape)
    # return (X > 0).astype('b') - 1*(X < 0).astype('b')

def rectified_linear_grad(X):
    return (X > 0).astype('b')
    
def softmax(X):
    """
    numpy.array -> numpy.array
    
    Compute softmax function: exp(X) / sum(exp(X))
    """
    mx = X.max()
    ex = exp(X.T - mx)
    return (ex / sum(ex,0)).T

def gaussian(X, mu=0, sigma=1):
    """
    (numpy.array, numpy.array, numpy.array) -> numpy.array
    
    Compute gaussian function: 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)**2/(2*sigma**2))
    """
    coef = 1/(sigma*sqrt(2*pi))
    return coef * exp(-(X-mu)**2 / (2*sigma**2)) 

def linear(X):
    return X

def rectified_linear(X):
    return X * (X > 0)

def sample_NRLU(X):
    sample = X + np.random.normal(size=X.shape)
    return rectified_linear(sample)

def sample_bernoulli(X):
    """
    numpy.array -> numpy.array
    
    All values of X must be probabilities of independent events occuring according
    to the binomial distribution
    
    Returns an indicator array of the same shape as input recording with
    output[i,j] = 1 iif X[i,j] >= uniform(0, 1)
    """
    return (X >= np.random.uniform(size=X.shape)).astype('b')

def sample_gaussian(X, mu=0, sigma=1):
    return X + np.random.normal(loc=mu, scale=sigma, size=X.shape)

def sample_softmax(X):
    """
    numpy.array -> numpy.array
    
    Returns an array with output[j, i] = 1 iif Xj = i is sampled
    """
    if len(np.shape(X)) == 1: X.shape = (1, len(X))
    num_points = np.shape(X)[0]
    output = np.zeros(np.shape(X))
    b = np.sum(np.cumsum(X.T,0) < rand(num_points), 0) - 1
    output[range(num_points),b] = 1
    return output
    
def cross_entropy(X, Y):
    return -np.sum((1 - Y) * np.log(1-X) + Y * np.log(X))

def cross_entropy_softmax(X, Y):
    return -np.sum(Y * np.log(X))
    
def square_error(X, Y):
    return np.sum(np.square(X - Y))
