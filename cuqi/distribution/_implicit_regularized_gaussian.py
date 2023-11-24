from cuqi.utilities import get_non_default_args
from cuqi.distribution import Distribution, Gaussian, GMRF
from cuqi.solver import ProjectNonnegative, ProjectBox, ProximalL1

import numpy as np


class ImplicitRegularizedGaussian(Distribution):
    """ Implicit Regularized Gaussian distribution.

    Defines a Gaussian distribution with implicit regularization. The regularization can be defined
    in the form of a proximal operator or a projector. Alternatively, preset constraints and regularization
    can be used.

    Precisely one of proximal, projector, constraint or regularization needs to be provided. Otherwise, an error is raised.

    Distribution can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.


    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.Gaussian` for details.

    cov
        See :class:`~cuqi.distribution.Gaussian` for details.

    prec
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtcov
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtprec
        See :class:`~cuqi.distribution.Gaussian` for details.

    proximal : callable f(x, scale) or None
        Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
        min_z 0.5||x-z||_2^2+scale*g(x).


    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints. Can be set to "nonnegativity" and "box". Required for use in Gibbs.
        For "box", the following additional parameters can be passed:
            lower_bound : array_like or None
                Lower bound of box, defaults to zero
            upper_bound : array_like
                Upper bound of box, defaults to one

    regularization : string or None
        Preset regularization. Can be set to "l1". Required for use in Gibbs in future update.


    """
        
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None, sqrtprec=None, proximal = None, projector = None, constraint = None, regularization = None, **kwargs):
        
        args = {"lower_bound" : kwargs.pop("lower_bound", None),
                "upper_bound" : kwargs.pop("upper_bound", None)}
        # We init the underlying Gaussian first for geometry and dimensionality handling
        self._gaussian = Gaussian(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov, sqrtprec=sqrtprec, **kwargs)

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, args)

    def _parse_regularization_input_arguments(self, proximal, projector, constraint, regularization, args):
        """ Parse regularization input arguments with guarding statements and store internal states """

        # Check that only one of proximal, projector, constraint or regularization is provided        
        if (proximal is not None) + (projector is not None) + (constraint is not None) + (regularization is not None) != 1:
            raise ValueError("Precisely one of proximal, projector, constraint or regularization needs to be provided.")

        if proximal is not None:
            if not callable(proximal):
                raise ValueError("Proximal needs to be callable.")
            if len(get_non_default_args(proximal)) != 2:
                raise ValueError("Proximal should take 2 arguments.")
            
        if projector is not None:
            if not callable(projector):
                raise ValueError("Projector needs to be callable.")
            if len(get_non_default_args(projector)) != 1:
                raise ValueError("Projector should take 1 argument.")
            
        # Preset information, for use in Gibbs
        self._preset = None
        
        if proximal is not None:
            self._proximal = proximal
        elif projector is not None:
            self._proximal = lambda z, gamma: projector(z)
        elif (isinstance(constraint, str) and constraint.lower().replace("-","") in ["nonnegativity", "nonnegative", "nn"]):
            self._proximal = lambda z, gamma: ProjectNonnegative(z)
            self._preset = "nonnegativity"
        elif (isinstance(constraint, str) and constraint.lower() in ["box"]):
            self._proximal = lambda z, gamma: ProjectBox(z, lower = args["lower_bound"] if "lower_bound" in args else None, upper = args["upper_bound"] if "upper_bound" in args else None)
            self._preset = "box" # Not supported in Gibbs
        elif (isinstance(regularization, str) and regularization.lower() in ["l1"]):
            self._proximal = ProximalL1
            self._preset = "l1"
        else:
            raise ValueError("Regularization not supported")

    # This is a getter only attribute for the underlying Gaussian
    # It also ensures that the name of the underlying Gaussian
    # matches the name of the implicit regularized Gaussian
    @property
    def gaussian(self):
        if self._name is not None:
            self._gaussian._name = self._name
        return self._gaussian
    
    @property
    def proximal(self):
        return self._proximal
    
    @property
    def preset(self):
        return self._preset

    def logpdf(self, x):
        return np.nan
        #raise ValueError(
        #    f"The logpdf of a implicit regularized Gaussian distribution need not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError(
            "There is no known way of efficiently sampling from a implicit regularized Gaussian distribution.")
  

    # --- Defer behavior of the underlying Gaussian --- #
    @property
    def geometry(self):
        return self.gaussian.geometry
    
    @geometry.setter
    def geometry(self, value):
        self.gaussian.geometry = value
    
    @property
    def mean(self):
        return self.gaussian.mean
    
    @mean.setter
    def mean(self, value):
        self.gaussian.mean = value
    
    @property
    def cov(self):
        return self.gaussian.cov
    
    @cov.setter
    def cov(self, value):
        self.gaussian.cov = value
    
    @property
    def prec(self):
        return self.gaussian.prec
    
    @prec.setter
    def prec(self, value):
        self.gaussian.prec = value
    
    @property
    def sqrtprec(self):
        return self.gaussian.sqrtprec
    
    @sqrtprec.setter
    def sqrtprec(self, value):
        self.gaussian.sqrtprec = value
    
    @property
    def sqrtcov(self):
        return self.gaussian.sqrtcov
    
    @sqrtcov.setter
    def sqrtcov(self, value):
        self.gaussian.sqrtcov = value     
    
    def get_conditioning_variables(self):
        return self.gaussian.get_conditioning_variables()
    
    def get_mutable_variables(self):
        return self.gaussian.get_mutable_variables()
    
    # Overwrite the condition method such that the underlying Gaussian is conditioned in general, except when conditioning on self.name
    # which means we convert Distribution to Likelihood or EvaluatedDensity.
    def _condition(self, *args, **kwargs):

        # Handle positional arguments (similar code as in Distribution._condition)
        cond_vars = self.get_conditioning_variables()
        kwargs = self._parse_args_add_to_kwargs(cond_vars, *args, **kwargs)

        # When conditioning, we always do it on a copy to avoid unintentional side effects
        new_density = self._make_copy()

        # Check if self.name is in the provided keyword arguments.
        # If so, pop it and store its value.
        value = kwargs.pop(self.name, None)

        new_density._gaussian = self.gaussian._condition(**kwargs)

        # If self.name was provided, we convert to a likelihood or evaluated density
        if value is not None:
            new_density = new_density.to_likelihood(value)

        return new_density
    

class ImplicitRegularizedGMRF(ImplicitRegularizedGaussian):
    """ Implicit Regularized GMRF (Gaussian Markov Random Field) distribution. 

    Defines a Gaussian distribution with implicit regularization. The regularization can be defined
    in the form of a proximal operator or a projector. Alternatively, preset constraints and regularization
    can be used.

    Only one of proximal, projector, constraint or regularization can be provided. If none of them are provided,
    a nonnegativity constraint is used by default.

    Distribution can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.


    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.GMRF` for details.
        
    prec
        See :class:`~cuqi.distribution.GMRF` for details.

    physical_dim
        See :class:`~cuqi.distribution.GMRF` for details.

    bc_type
        See :class:`~cuqi.distribution.GMRF` for details.

    order
        See :class:`~cuqi.distribution.GMRF` for details.

    proximal : callable f(x, scale) or None
        Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
        min_z 0.5||x-z||_2^2+scale*g(x).

    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints, including "nonnegativity" and "box". Required for use in Gibbs.

    regularization : string or None
        Preset regularization, including "l1". Required for use in Gibbs in future update.

    """
    # TODO: Once GMRF is updated, add default None to mean and prec here.
    def __init__(self, mean, prec, physical_dim=1, bc_type='zero', order=1, proximal = None, projector = None, constraint = None, regularization = None, **kwargs):
            
            args = {"lower_bound" : kwargs.pop("lower_bound", None),
                    "upper_bound" : kwargs.pop("upper_bound", None)}
            
            # Underlying explicit Gaussian
            self._gaussian = GMRF(mean, prec, physical_dim=physical_dim, bc_type=bc_type, order=order, **kwargs)
            
            # Init from abstract distribution class
            super(Distribution, self).__init__(**kwargs)

            self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, args)
