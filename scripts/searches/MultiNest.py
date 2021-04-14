"""
Searches: MultiNest
=======================

This example illustrates how to use the nested sampling algorithm MultiNest.

Information about MultiNest can be found at the following links:

 - https://github.com/JohannesBuchner/MultiNest
 - https://github.com/JohannesBuchner/PyMultiNest
 - http://johannesbuchner.github.io/PyMultiNest/index.html#
"""
import autofit as af
import model as m
import analysis as a

import matplotlib.pyplot as plt
import numpy as np
from os import path

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
model = af.PriorModel(m.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = a.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We now create and run the `MultiNest` object which acts as our non-linear search. 

We manually specify all of the MultiNest settings, descriptions of which are provided at the following webpage:

 - https://github.com/JohannesBuchner/MultiNest
 - https://github.com/JohannesBuchner/PyMultiNest
 - http://johannesbuchner.github.io/PyMultiNest/index.html#
"""
multi_nest = af.MultiNest(
    path_prefix="searches",
    name="MultiNest",
    nlive=50,
    n_live_points=50,
    sampling_efficiency=0.2,
    const_efficiency_mode=False,
    evidence_tolerance=0.5,
    multimodal=False,
    importance_nested_sampling=False,
    max_modes=100,
    mode_tolerance=-1e90,
    max_iter=0,
    n_iter_before_update=5,
    null_log_evidence=-1e90,
    seed=0,
    verbose=False,
    resume=True,
    context=0,
    write_output=True,
    log_zero=-1e100,
    init_MPI=False,
    iterations_per_update=500,
    number_of_cores=2,
)

result = multi_nest.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.profile_from_xvalues(
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
plt.title("MultiNest model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()
