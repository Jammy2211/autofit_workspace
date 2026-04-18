"""
Searches: Maximum Likelihood Estimation
=======================================

This example illustrates how to use the maximum likelihood / optimization algorithms supported by **PyAutoFit**:

 - `Drawer`: Draws a fixed number of samples uniformly from the priors (useful for sensitivity mapping and
   quantifying stochasticity of the likelihood function).
 - `LBFGS`: The scipy L-BFGS-B optimization algorithm.

Relevant links:

 - Drawer: https://github.com/rhayes777/PyAutoFit/blob/main/autofit/non_linear/optimize/drawer/drawer.py
 - L-BFGS: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

__Contents__

This script is split into the following sections:

- **Data**: Loading and plotting the 1D Gaussian dataset used to demonstrate the searches.
- **Model + Analysis**: Setting up the model and analysis shared by every search below.
- **Search: Drawer**: Configuring and running the Drawer search.
- **Search: LBFGS**: Configuring and running the L-BFGS-B optimizer.
- **Search Internal**: Accessing the internal optimizer for advanced use (shown once for LBFGS).
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

This model and analysis are shared by every MLE search below. `use_jax=False` is required by `LBFGS` and is
harmless for `Drawer`.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=False)

"""
__Search: Drawer__

We now create and run the `Drawer` object which acts as our non-linear search.

The `Drawer` simply draws a fixed number of samples from the model uniformly from the priors. It does not
seek to determine model parameters which maximize the likelihood or map out the posterior of the overall
parameter space.

Whilst this is not the typical use case of a non-linear search, it has certain niche applications, for example:

 - Given a model one can determine how much variation there is in the log likelihood / log posterior values.
 By visualizing this as a histogram one can therefore quantify the behaviour of that
 model's `log_likelihood_function`.

 - If the `log_likelihood_function` of a model is stochastic (e.g. different values of likelihood may be
 computed for an identical model due to randomness in the likelihood evaluation) this search can quantify
 the behaviour of that stochasticity.

 - For advanced modeling tools, for example sensitivity mapping performed via the `Sensitivity` object,
 the `Drawer` search may be sufficient to perform the overall modeling task, without the need of performing
 an actual parameter space search.
"""
search = af.Drawer(path_prefix="searches", name="Drawer", total_draws=3)

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
plt.title("Drawer model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search: LBFGS__

We now create and run the `LBFGS` object which acts as our non-linear search.

We manually specify all of the LBFGS settings, descriptions of which are provided at the following webpage:

 https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
"""
search = af.LBFGS(
    path_prefix="searches",
    name="LBFGS",
    tol=None,
    disp=None,
    maxcor=10,
    ftol=2.220446049250313e-09,
    gtol=1e-05,
    eps=1e-08,
    maxfun=15000,
    maxiter=15000,
    iprint=-1,
    maxls=20,
    iterations_per_full_update=1000,
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
plt.title("LBFGS model fit to 1D Gaussian dataset.")
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

For `LBFGS`, this is the scipy `OptimizeResult` object returned by `scipy.optimize.minimize` with ``method="L-BFGS-B"``.

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
