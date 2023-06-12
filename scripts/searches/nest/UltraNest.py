"""
Searches: UltraNest
=======================

This example illustrates how to use the nested sampling algorithm UltraNest.

UltraNest is an optional requirement and must be installed manually via the command `pip install ultranest`.
It is optional as it has certain dependencies which are generally straight forward to install (e.g. Cython).

Information about Dynesty can be found at the following links:

 - https://github.com/JohannesBuchner/UltraNest
 - https://johannesbuchner.github.io/UltraNest/readme.html
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

We now create and run the `UltraNest` object which acts as our non-linear search. 

We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:

- https://johannesbuchner.github.io/UltraNest/readme.html
- https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler

"""
search = af.UltraNest(
    path_prefix="searches",
    name="UltraNest7",
    resume=True,
    run_num=None,
    num_test_samples=2,
    draw_multiple=True,
    num_bootstraps=30,
    vectorized=False,
    ndraw_min=128,
    ndraw_max=65536,
    storage_backend="hdf5",
    warmstart_max_tau=-1,
    update_interval_volume_fraction=0.8,
    update_interval_ncall=None,
    log_interval=None,
    show_status=True,
    viz_callback="auto",
    dlogz=0.5,
    dKL=0.5,
    frac_remain=0.01,
    Lepsilon=0.001,
    min_ess=400,
    max_iters=None,
    max_ncalls=None,
    max_num_improvement_loops=-1,
    min_num_live_points=50,
    cluster_num_live_points=40,
    insertion_test_window=10,
    insertion_test_zscore_threshold=2,
    stepsampler_cls="RegionMHSampler",
    nsteps=11,
    number_of_cores=1,
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
plt.title("UltraNest model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()
