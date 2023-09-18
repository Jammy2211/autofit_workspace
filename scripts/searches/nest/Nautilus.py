"""
Searches=Nautilus
=======================

This example illustrates how to use the nested sampling algorithm Nautilus.

Information about Nautilus can be found at the following links:

 - https://nautilus-sampler.readthedocs.io/en/stable/index.html
 - https://github.com/johannesulf/nautilus
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

"""
__Data__

This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
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
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

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
    n_shell=None,  # Minimum number of points in each shell. The algorithm will sample from the shells until this is reached. Default is the batch size of the sampler which is 100 unless otherwise specified.
    n_eff=500,  # Minimum effective sample size. The algorithm will sample from the shells until this is reached. Default is 10000.
    discard_exploration=False,  # Whether to discard points drawn in the exploration phase. This is required for a fully unbiased posterior and evidence estimate.
    verbose=True,  # Whether to print information about the run.
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
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
