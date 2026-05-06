"""
Searches: MCMC
==============

This example illustrates how to use the MCMC sampler algorithms supported by **PyAutoFit**:

 - `Emcee`: The emcee ensemble sampler.
 - `Zeus`: The zeus ensemble slice sampler.
 - `BlackJAXNUTS`: BlackJAX's No-U-Turn Sampler — gradient-based MCMC requiring `use_jax=True`.

Relevant links:

 - Emcee:    https://emcee.readthedocs.io/en/stable/
 - Zeus:     https://zeus-mcmc.readthedocs.io/en/latest/
 - BlackJAX: https://github.com/blackjax-devs/blackjax

__Contents__

This script is split into the following sections:

- **Data**: Loading and plotting the 1D Gaussian dataset used to demonstrate the searches.
- **Model + Analysis**: Setting up the model and analysis shared by every search below.
- **Search: Emcee**: Configuring and running the Emcee sampler.
- **Search: Zeus**: Configuring and running the Zeus sampler.
- **Search: BlackJAXNUTS**: Configuring and running BlackJAX's NUTS sampler (requires `use_jax=True`).
- **Search Internal**: Accessing the internal sampler for advanced use (shown once for Emcee).
"""

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

"""
__Data__

This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.show()
plt.close()

"""
__Model + Analysis__

We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.

This model and analysis are shared by every MCMC sampler below.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search: Emcee__

We now create and run the `Emcee` object which acts as our non-linear search.

We manually specify all of the Emcee settings, descriptions of which are provided at the following webpage:

 https://emcee.readthedocs.io/en/stable/user/sampler/
 https://emcee.readthedocs.io/en/stable/
"""
search = af.Emcee(
    path_prefix="searches",
    name="Emcee",
    nwalkers=30,
    nsteps=1000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.model_data_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("Emcee model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search Internal__

The result also contains the internal representation of the non-linear search.

The internal representation of the non-linear search ensures that all sampling info is available in its native form.
This can be passed to functions which take it as input, for example if the sampling package has bespoke visualization
functions.

For `Emcee`, this is an instance of the `Sampler` object (`from emcee import EnsembleSampler`).
For `Zeus`, this is an instance of `EnsembleSampler` from the `zeus` package.

The internal search is by default not saved to hard-disk, because it can often take up quite a lot of hard-disk space
(significantly more than standard output files).

This means that the search internal will only be available the first time you run the search. If you rerun the code
and the search is bypassed because the results already exist on hard-disk, the search internal will not be available.

If you are frequently using the search internal you can have it saved to hard-disk by changing the `search_internal`
setting in `output.yaml` to `True`. The result will then have the search internal available as an attribute,
irrespective of whether the search is re-run or not.
"""
search_internal = result.search_internal

print(search_internal)

"""
__Search: Zeus__

We now create and run the `Zeus` object which acts as our non-linear search.

We manually specify all of the Zeus settings, descriptions of which are provided at the following webpage:

 https://zeus-mcmc.readthedocs.io/en/latest/
 https://zeus-mcmc.readthedocs.io/en/latest/api/sampler.html
"""
search = af.Zeus(
    path_prefix="searches",
    name="Zeus",
    nwalkers=30,
    nsteps=1001,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    tune=False,
    tolerance=0.05,
    patience=5,
    maxsteps=10000,
    mu=1.0,
    maxiter=10000,
    vectorize=False,
    check_walkers=True,
    shuffle_ensemble=True,
    light_mode=False,
    iterations_per_full_update=501,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

model_data = result.max_log_likelihood_instance.model_data_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("Zeus model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search: BlackJAXNUTS__

`BlackJAXNUTS` is the No-U-Turn Sampler from BlackJAX — a gradient-based MCMC that extends
Hamiltonian Monte Carlo by adapting trajectory length on the fly, so the user does not have to
hand-tune the leapfrog step count. Because it uses gradients, the run is typically much more
sample-efficient than the ensemble samplers above on smooth, unimodal posteriors.

Two requirements distinguish it from `Emcee` / `Zeus`:

 1) The analysis must be built with `use_jax=True` so the log-likelihood is JAX-traceable end to
    end (`jax.grad` of it has to be takeable). Below we construct a separate `analysis_jax` to
    keep the existing `analysis` (used by Emcee / Zeus) on its NumPy path.
 2) The user's model must be registered as a JAX pytree so `model.instance_from_vector` flows
    through `jax.jit`. `enable_pytrees()` is a one-shot process-level call; `register_model(model)`
    walks the user's model and registers each concrete class it finds (here, `af.ex.Gaussian`).
    This is the same mechanism the `Nautilus_jax` example uses.

The fit itself runs in two phases:

 1) `blackjax.window_adaptation` (warmup) — tunes the leapfrog step size via dual averaging and
    a diagonal inverse mass matrix from the warmup covariance.
 2) NUTS sampling with the tuned kernel, run inside a JIT-compiled `jax.lax.scan` so the inner
    per-step kernel runs as a single fused XLA computation.

`blackjax` is an optional dependency — install with `pip install autofit[optional]` (which now
pulls it in alongside `nautilus-sampler` etc.) or directly with `pip install blackjax`.

Relevant links:

 - BlackJAX:                  https://github.com/blackjax-devs/blackjax
 - BlackJAX docs:             https://blackjax-devs.github.io/blackjax/
 - The No-U-Turn paper:       https://arxiv.org/abs/1111.4246

If you use `BlackJAXNUTS` as part of a published work, please cite the BlackJAX package following
the instructions on its GitHub page.
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

analysis_jax = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

search = af.BlackJAXNUTS(
    path_prefix="searches",
    name="BlackJAXNUTS",
    num_warmup=500,
    num_samples=1000,
    target_accept=0.8,
)

result = search.fit(model=model, analysis=analysis_jax)

model_data = result.max_log_likelihood_instance.model_data_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("BlackJAXNUTS model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
The `samples_info` dict on `result.samples` exposes NUTS-specific diagnostics that don't apply to
the ensemble samplers above:

 - `ess_min` / `ess_per_param`: effective sample size from BlackJAX's Geyer-style estimator.
   Higher is better; values close to `num_samples` mean the chain is nearly independent.
 - `mean_acceptance`: average Metropolis acceptance over the post-warmup chain. Should land
   close to `target_accept` (0.8 by default) once warmup has converged.
 - `n_divergent`: number of divergent transitions post-warmup. A non-zero count usually means
   the step size is too aggressive or the posterior has narrow funnels — re-run with a
   tighter `target_accept` (e.g. 0.95) if you see them.
 - `n_logl_evals`: total leapfrog integration steps summed across the chain — the right
   "cost-per-sample" denominator for comparing NUTS to ensemble methods.
"""
info = result.samples.samples_info
print(f"ESS (min over dims):   {info['ess_min']:.1f} / {info['num_samples']}")
print(f"Mean acceptance:       {info['mean_acceptance']:.3f} (target {0.8:.2f})")
print(f"Divergences:           {info['n_divergent']}")
print(f"Total leapfrog evals:  {info['n_logl_evals']}")
