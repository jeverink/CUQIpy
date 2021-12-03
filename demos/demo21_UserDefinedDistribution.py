#%%
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
import cuqi

from scipy.sparse import diags

#%% Create X ~ Normal( -1.0,4.0) distirbution

mu1 = -1.0
std1 = 4.0
N = cuqi.distribution.Normal(mean=mu1, std=std1)

dim1 = 1


#%%
logpdf_func = lambda xx: -np.log(std1*np.sqrt(2*np.pi))-0.5*((xx-mu1)/std1)**2
NU = cuqi.distribution.UserDefinedDistribution(dim=dim1, logpdf_func=logpdf_func)

#%% Plot PDF
grid = np.linspace(-10.0, 10.0, 101)
plt.plot(grid, N.logpdf(grid), LineWidth=3)
plt.plot(grid, NU.logpdf(grid), '--')

#%% Compare samples
N.sample()

#%%
NU.sample()

#%%
NU.sample_func = lambda : mu1 + std1*np.random.randn(dim1,1)

#%%
NU.sample()

#%%
NS = 10000
s_N  = N.sample(NS)
s_NU = NU.sample(NS)

s_N.hist_chain(0, bins=100)
s_NU.hist_chain(0, bins=100)

#%%  NOW DO THE SAME TEST FOR A 5-dim GAUSSIANCOV

#%%
mu2 = np.array([1,2,3,4,5])
cov2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

dim2 = 5

G = cuqi.distribution.GaussianCov(mean=mu2, cov=cov2)

#%%

logpdf_func2 = lambda xx: -0.5*(dim2*np.log(2*np.pi) + np.sum(np.log(cov2)) + \
        np.sum(np.square( (xx-mu2) @ diags(np.sqrt(1/cov2))), axis=-1))
GU = cuqi.distribution.UserDefinedDistribution(logpdf_func=logpdf_func2)

#%%
try_point = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

G.logpdf(try_point)

#%%
GU.logpdf(try_point)
# %%
