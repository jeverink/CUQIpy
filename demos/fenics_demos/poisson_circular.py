# %%
from turtle import pos
from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi
from mshr import *
import matplotlib.pyplot as plt

#%%
class matern():
    def __init__(self, path, num_terms=128):
        self.dim = num_terms
        matern_data = np.load(path)
        self.eig_val = matern_data['l'][:num_terms]
        self.eig_vec = matern_data['e'][:,:num_terms]

    def set_levels(self, c_minus=0., c_plus=1.):
        self.c_minus = c_minus
        self.c_plus = c_plus

    def heavy(self, x):
        return self.c_minus*0.5*(1 + np.sign(x)) + self.c_plus*0.5*(1 - np.sign(x))

    def assemble(self, p):
        return self.heavy(self.eig_vec@( self.eig_val*p ))

class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)) )

def u_boundary(x, on_boundary):
    return False

#obs_func = lambda m,u : u.split()[0]
obs_func = None

domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain, 20)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)
V_space = FunctionSpace(mesh, V)

FEM_el = parameter_space.ufl_element()
source_term = source(element=FEM_el)

matern_field = matern('basis.npz')
matern_field.set_levels(1,10)
#m_func = Function( parameter_space )
def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]

    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds - source_term*v_0*dx

bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)


PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

#%%
domain_geometry = cuqi.fenics.geometry.FEniCSMatern(parameter_space, matern_field)

range_geometry = cuqi.fenics.geometry.FEniCSContinuous(solution_space) 

m_input = cuqi.samples.CUQIarray( np.random.standard_normal(128), geometry= domain_geometry)


PDE.assemble(m_input)
sol, _ = PDE.solve()
observed_sol = PDE.observe(sol)

plot(sol[0])

#%%
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

#%%
# Create prior
pr_mean = np.zeros(domain_geometry.dim)
prior = cuqi.distribution.GaussianCov(pr_mean, cov=np.eye(domain_geometry.dim), geometry= domain_geometry)


# Exact solution
exactSolution = prior.sample()

# Exact data
b_exact = model.forward(domain_geometry.par2fun(exactSolution),is_par=False)

# %%
# Add noise to data
SNR = 100
sigma = np.linalg.norm(b_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise
data = b_exact + np.random.normal( 0, sigma, b_exact.shape )

# Create likelihood
#likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(range_geometry.dim)).to_likelihood(data)
likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.ones(range_geometry.dim)).to_likelihood(data)

posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% MH Sampler
MHSampler = cuqi.sampler.MetropolisHastings(
    posterior,
    proposal=None,
    scale=None,
    x0=None,
    dim=None,
)

samples = MHSampler.sample_adapt(1000)



#%%
plt.figure()
im = plot(domain_geometry.par2fun(exactSolution), title="exact solution")
plt.colorbar(im)

# %%
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])

# %%
ims = samples.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

# %%
samples.plot_trace()
samples.plot_autocorrelation(max_lag=300)


# %% 
pCNSampler = cuqi.sampler.pCN(
    posterior,
    scale=None,
    x0=None,
)

samplespCN = pCNSampler.sample_adapt(1000)


#%%
plt.figure()
im = plot(domain_geometry.par2fun(exactSolution), title="exact solution")
plt.colorbar(im)

# %%
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])

# %%
ims = samplespCN.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

# %%
samplespCN.plot_trace()
samplespCN.plot_autocorrelation(max_lag=300)

# %%
plt.figure()
samples.plot_ci(plot_par = True)
plt.title("Credible interval MH")
# %%
