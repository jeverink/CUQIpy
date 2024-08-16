# Load modules and set-up test problem
import sys
sys.path.append("..")

import pickle

import numpy as np
import matplotlib.pyplot as plt

import cuqi
from cuqi.geometry import Continuous1D, Continuous2D
from cuqi.testproblem import Deconvolution1D, Deconvolution2D
from cuqi.distribution import Gaussian, Gamma, ModifiedHalfNormal, JointDistribution, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF, RegularizedUniform
from cuqi.sampler import LinearRTO, RegularizedLinearRTO, Gibbs, Conjugate
from cuqi.model import LinearModel

# Set seed
np.random.seed(24601)

# Create a new sampler class with preset parameters
def make_RegularizedLinearRTO(x0 = None, maxit = 100, stepsize = "automatic", tradeoff = 10.0, adaptive = True):
    # Wrapper to tune the parameters
    class _RegularizedLinearRTO(RegularizedLinearRTO):
        def __init__(self, target):
            super().__init__(target, x0=x0, maxit=maxit, stepsize = stepsize, tradeoff = tradeoff, adaptive = adaptive)

    return _RegularizedLinearRTO


file_name = "sample_data//samples_TVMHN_box.dat"
def save_samples(samples):
    with open(file_name, "wb+") as file:
        pickle.dump(samples, file)
        file.close()

def load_samples():
    with open(file_name, "rb") as file:
        samples = pickle.load(file)
        file.close()
    return samples


## Create problem
from generate_data import gen_problem

n = 128
num_samples, num_burnin = 10000, 100

A_mat, y_data, exactSolution = gen_problem()
A = LinearModel(A_mat, domain_geometry=Continuous1D(128), range_geometry=Continuous1D(128))

# Reproducability
np.random.seed(24601)

# Create posterior
l = Gamma(1, 1e-4)
d = ModifiedHalfNormal(1, 1e-4, -1e-4) #Gamma(1, 1e-4)
x = RegularizedGMRF(mean = np.zeros(n), prec = lambda d : 2.4e-4*d**2,
                    regularization = "TV", strength = lambda d : d,
                    constraint = "box", lower_bound = 0.0, upper_bound = 0.0,
                    geometry = A.domain_geometry,)
y = Gaussian(A@x, prec = lambda l : l)

joint = JointDistribution(l, d, x, y)
posterior = joint(y=y_data)

# Sample

sampling_strategy = {
    'x': make_RegularizedLinearRTO(x0=None, maxit=100, tradeoff = 10.0, adaptive = False),
    'l': Conjugate,
    'd': Conjugate,
}

sampler = Gibbs(posterior, sampling_strategy,
                sampling_order = ['x', 'l', 'd'],
                initial_guess = {'l': 100, 'd':100})
samples = sampler.sample(num_samples, num_burnin)

# Save samples
save_samples(samples)