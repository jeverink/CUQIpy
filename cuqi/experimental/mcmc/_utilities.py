import cuqi
import inspect
import numpy as np

# This import makes suggest_sampler easier to read
import cuqi.experimental.mcmc as samplers 

def find_valid_samplers(target, as_string = True):
    """ Finds all samplers in the cuqi.experimental.mcmc module that accept the provided target. """

    all_samplers = [(name, cls) for name, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass) if issubclass(cls, cuqi.experimental.mcmc.Sampler)]
    valid_samplers = []

    for name, sampler in all_samplers:
        try:
            sampler(target)
            valid_samplers += [name if as_string else sampler]
        except:
            pass

    # Need a separate case for HybridGibbs
    if find_valid_sampling_strategy(target) is not None:
        valid_samplers += [cuqi.experimental.mcmc.HybridGibbs.__name__ if as_string else cuqi.experimental.mcmc.HybridGibbs]

    return valid_samplers

def find_valid_sampling_strategy(target, as_string = True):
    """
        Find valid samplers to be used for creating a sampling strategy for the HybridGibbs sampler
    """

    if not isinstance(target, cuqi.distribution.JointDistribution):
        return None

    par_names = target.get_parameter_names()

    valid_samplers = dict()
    for par_name in par_names:
        conditional_params = {par_name_: np.ones(target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
        conditional = target(**conditional_params)

        samplers = find_valid_samplers(conditional, as_string)
        if len(samplers) == 0:
            return None
        
        valid_samplers[par_name] = samplers

    return valid_samplers

def suggest_sampler(target, as_string = False, exceptions = []):
    """
        Suggests a possible sampler that can be used for sampling from the target distribution.

        DESCRIBE ARGUMENTS BEHIND SUGGESTION
    """

    n = target.dim

    # Samplers with suggested default values (when no defaults are defined)
    ordering = [
        # Direct and Conjugate samplers
        (samplers.Direct, {}),
        (samplers.Conjugate, {}),
        (samplers.ConjugateApprox, {}),
        # Specialized samplers
        (samplers.LinearRTO, {}),
        (samplers.RegularizedLinearRTO, {}),
        (samplers.UGLA, {}),
        # Hamiltonian samplers
        (samplers.NUTS, {}),
        # Langevin based samplers
        (samplers.MALA, {}),
        (samplers.ULA, {}),
        # Gibbs and Componentwise samplers
        (samplers.HybridGibbs, {}),
        (samplers.CWMH, {"scale" : 0.05*np.ones(n),
                         "x0" : 0.5*np.ones(n)}),
        # Proposal based samplers
        (samplers.PCN, {"scale" : 0.02}),
        (samplers.MH, {}),
    ]

    valid_samplers = find_valid_samplers(target, as_string = False)
    
    for suggestion, values in ordering:
        if suggestion in valid_samplers and suggestion not in exceptions:
            # Sampler found
            if as_string:
                return suggestion.__name__
            else:
                # Cases for samplers that might need default values
                if suggestion is not samplers.HybridGibbs:
                    return suggestion(target, **values)
                else:
                    return suggestion(target, sampling_strategy = suggest_sampling_strategy(target, as_string = False))

            
    return None

def suggest_sampling_strategy(target, as_string = False):
    """
        Suggests a possible sampling strategy to be used with the HybridGibbs sampler.

        For reasoning behind the suggestion, see 'suggest_sampler'.
    """

    if not isinstance(target, cuqi.distribution.JointDistribution):
        return None

    par_names = target.get_parameter_names()

    suggested_samplers = dict()
    for par_name in par_names:
        conditional_params = {par_name_: np.ones(target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
        conditional = target(**conditional_params)

        sampler = suggest_sampler(conditional, as_string = as_string)
        if sampler is None:
            return None
        
        suggested_samplers[par_name] = sampler

    return suggested_samplers