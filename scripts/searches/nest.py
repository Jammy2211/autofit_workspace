"""
Searches: Nested Sampling
=========================

This example illustrates how to use the nested sampling algorithms supported by **PyAutoFit**:

 - `DynestyStatic`: Dynesty with static nested sampling.
 - `DynestyDynamic`: Dynesty with dynamic nested sampling.
 - `Nautilus`: Nautilus nested sampler.
 - `NSS`: Nested Slice Sampling (JAX-native, optional install).

The first three are boundary-based samplers that work with any Python log-likelihood. `NSS` is a more recent
JAX-native sampler that runs its inner sampling loop inside `jax.jit` — when the log-likelihood is itself
JAX-traceable, the per-evaluation cost drops by roughly an order of magnitude versus the boundary samplers
(measured on real lensing likelihoods, see the `nss_first_class_sampler` roadmap and FINDINGS_v3 in the
profiling project for the numbers).

Relevant links:

 - Dynesty: https://dynesty.readthedocs.io/en/latest/
 - Nautilus: https://nautilus-sampler.readthedocs.io/en/stable/
 - NSS (Nested Slice Sampling): https://github.com/yallup/nss

__Install Precondition for NSS__

The `Search: NSS` section at the bottom of this script imports the optional `nss` package. To run that
section, install the dependencies first:

    pip install autofit[nss]

The extra pins the right `handley-lab/blackjax` fork at a known-good commit, so this is a single safe
command (no `--no-deps` dance, no manual git+ URLs). The other three samplers in this script have no
additional dependencies and run with the standard `pip install autofit` install.

__Contents__

This script is split into the following sections:

- **Data**: Loading and plotting the 1D Gaussian dataset used to demonstrate the searches.
- **Model + Analysis**: Setting up the model and analysis shared by every search below.
- **Search: DynestyStatic**: Configuring and running the DynestyStatic nested sampler.
- **Search: DynestyDynamic**: Configuring and running the DynestyDynamic nested sampler.
- **Search: Nautilus**: Configuring and running the Nautilus nested sampler.
- **Search: NSS**: Configuring and running the NSS (Nested Slice Sampling) sampler.
- **Search Internal**: Accessing the internal sampler for advanced use (shown once for DynestyStatic).
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

This model and analysis are shared by every nested sampler below.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search: DynestyStatic__

We now create and run the `DynestyStatic` object which acts as our non-linear search.

We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:

 https://dynesty.readthedocs.io/en/latest/api.html
 https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.nestedsamplers
"""
search = af.DynestyStatic(
    path_prefix=path.join("searches"),
    name="DynestyStatic",
    nlive=50,
    bound="multi",
    sample="auto",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    iterations_per_full_update=2500,
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
plt.title("DynestyStatic model fit to 1D Gaussian dataset.")
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

For `DynestyStatic`, this is an instance of the `NestedSampler` object (`from dynesty import NestedSampler`).

The internal search is by default not saved to hard-disk, because it can often take up quite a lot of hard-disk space
(significantly more than standard output files).

This means that the search internal will only be available the first time you run the search. If you rerun the code
and the search is bypassed because the results already exist on hard-disk, the search internal will not be available.

If you are frequently using the search internal you can have it saved to hard-disk by changing the `search_internal`
setting in `output.yaml` to `True`. The result will then have the search internal available as an attribute,
irrespective of whether the search is re-run or not.

The equivalent internal sampler for `DynestyDynamic` is `DynamicNestedSampler` and for `Nautilus` is `Sampler`.
"""
search_internal = result.search_internal

print(search_internal)

"""
__Search: DynestyDynamic__

We now create and run the `DynestyDynamic` object which acts as our non-linear search.

We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:

 https://dynesty.readthedocs.io/en/latest/api.html
 https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.dynamicsampler
"""
search = af.DynestyDynamic(
    path_prefix="searches",
    name="DynestyDynamic",
    nlive=50,
    bound="multi",
    sample="auto",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
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
plt.title("DynestyDynamic model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search: Nautilus__

We now create and run the `Nautilus` object which acts as our non-linear search.

We manually specify all of the Nautilus settings, descriptions of which are provided at the following webpage:

 https://github.com/johannesulf/nautilus
"""
search = af.Nautilus(
    path_prefix=path.join("searches"),
    name="Nautilus",
    number_of_cores=4,
    n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
    n_update=None,  # The maximum number of additions to the live set before a new bound is created
    enlarge_per_dim=1.1,  # Along each dimension, outer ellipsoidal bounds are enlarged by this factor.
    n_points_min=None,  # The minimum number of points each ellipsoid should have. Effectively, ellipsoids with less than twice that number will not be split further.
    split_threshold=100,  # Threshold used for splitting the multi-ellipsoidal bound used for sampling.
    n_networks=4,  # Number of networks used in the estimator.
    n_batch=100,  # Number of likelihood evaluations that are performed at each step. If likelihood evaluations are parallelized, should be multiple of the number of parallel processes.
    n_like_new_bound=None,  # The maximum number of likelihood calls before a new bounds is created. If None, use 10 times n_live.
    vectorized=False,  # If True, the likelihood function can receive multiple input sets at once.
    seed=None,  # Seed for random number generation used for reproducible results accross different runs.
    f_live=0.01,  # Maximum fraction of the evidence contained in the live set before building the initial shells terminates.
    n_shell=1,  # Minimum number of points in each shell. The algorithm will sample from the shells until this is reached. Default is 1.
    n_eff=500,  # Minimum effective sample size. The algorithm will sample from the shells until this is reached. Default is 10000.
    discard_exploration=False,  # Whether to discard points drawn in the exploration phase. This is required for a fully unbiased posterior and evidence estimate.
    verbose=True,  # Whether to print information about the run.
    n_like_max=np.inf,  # Maximum number of likelihood evaluations. Regardless of progress, the sampler will stop if this value is reached. Default is infinity.
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
plt.title("Nautilus model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search: NSS__

NSS (Nested Slice Sampling) is a JAX-native nested sampler whose inner sampling loop runs end-to-end inside
`jax.jit`. The advantage versus the boundary samplers above is **per-evaluation cost**: when your
log-likelihood is itself JAX-traceable, NSS avoids the Python ↔ JAX boundary that Nautilus and Dynesty cross
on every likelihood call. On the production lensing likelihoods that motivated this sampler, the per-eval
cost is roughly 30 times lower than Nautilus's, and total wall time to convergence drops from tens of
minutes to a few minutes.

On the trivial 1D Gaussian dataset used by this tutorial the speedup is not visible — the likelihood is so
cheap that the per-call cost is dominated by Python overhead rather than the floating-point work, and the
first NSS run pays a one-off ~25–30 second JIT compile that the simpler samplers above skip. The numbers
below let you confirm NSS converged to roughly the same posterior, not that it ran faster. Try NSS on a
real autolens or autogalaxy MGE / pixelization likelihood to see the per-eval advantage.

NSS exposes the same `result.samples` interface as the boundary samplers — swapping `af.Nautilus(...)` for
`af.NSS(...)` in your existing scripts is a one-line change.

A few NSS-specific kwargs to be aware of:

 - `n_live`: live particles maintained throughout the run (production default 200).
 - `num_mcmc_steps`: slice-MCMC inner steps per dead-point batch (production default 5).
 - `num_delete`: particles removed per outer iteration (production default 50; larger values reduce JIT
   overhead per outer iteration at the cost of slightly coarser posterior coverage).
 - `termination`: stopping criterion on `logZ_live - logZ` (default `-3.0`, corresponding to a remaining
   evidence fraction below 1e-3).
 - `checkpoint_interval`: outer iterations between disk-saved checkpoints. NSS writes a resumable state
   file every `checkpoint_interval` iterations, so a SLURM timeout halfway through a long fit is recovered
   automatically the next time you run the same script with the same Paths.
 - `iterations_per_quick_update`: when set non-None, NSS calls `analysis.visualize(...)` with the current
   best live point every N outer iterations — partial results appear in the image_path directory while the
   run is still in flight.

Settings reference: see `af.NSS.__init__` for the full kwarg list.

__Optional Dependency Guard__

The standard workspace release environment does not install `autofit[nss]`. When the optional stack is
absent, this script skips only the NSS fit and leaves the Dynesty / Nautilus examples above runnable.

__Analysis Must Be JAX-Traceable__

NSS runs the log-likelihood inside `jax.jit`. The boundary-based samplers above are happy with the default
NumPy `af.ex.Analysis(data, noise_map)` — when we hand the same analysis to NSS the JIT trace hits the
NumPy paths inside the analysis and raises `TracerArrayConversionError`. The fix is to build the analysis
with `use_jax=True`, which makes its internal arithmetic dispatch through `jax.numpy` instead of `numpy`.

This is the production pattern: for autolens / autogalaxy / autofit analyses that you want to run with NSS,
construct your `Analysis` with `use_jax=True`. Everything below works identically to the NumPy path — same
`log_likelihood_function` API, same `Result` shape — but the body is now JAX-traceable.
"""
try:
    search = af.NSS(
        path_prefix=path.join("searches"),
        name="NSS",
        n_live=200,  # live particles maintained throughout the run
        num_mcmc_steps=5,  # slice-MCMC inner steps per dead-point batch
        num_delete=50,  # particles removed per outer iteration
        termination=-3.0,  # delta-logZ stopping criterion
        seed=42,  # JAX PRNG seed for reproducible runs
        checkpoint_interval=100,  # SLURM-friendly resume — see docstring above
    )
except ImportError as exc:
    print(f"Skipping NSS example because the optional dependency stack is unavailable:\n{exc}")
else:
    analysis_jax = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

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
    plt.title("NSS model fit to 1D Gaussian dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

    print(f"NSS log evidence:    {result.samples.samples_info['log_evidence']:.4f}")
    print(f"NSS max log L:       {max(result.samples.log_likelihood_list):.4f}")
