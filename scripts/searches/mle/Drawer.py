"""
Searches: Drawer
================

This example illustrates how to use the Drawer search, simply draws a fixed number of samples from the model uniformly
from the priors.

Therefore, it does not seek to determine model parameters which maximize the likelihood or map out the
posterior of the overall parameter space.

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
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We now create and run the `Drawer` object which acts as our non-linear search. 

We manually specify all of the Drawer settings, descriptions of which are provided at the following webpage:

https://github.com/rhayes777/PyAutoFit/blob/main/autofit/non_linear/optimize/drawer/drawer.py
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
plt.title("PySwarmsLocal model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()
