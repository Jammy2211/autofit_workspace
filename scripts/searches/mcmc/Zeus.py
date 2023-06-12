"""
Searches: DynestyStatic
=======================

This example illustrates how to use the MCMC ensamble sampler algorithm Zeus.

Information about Zeus can be found at the following links:

 - https://github.com/minaskar/zeus
 - https://zeus-mcmc.readthedocs.io/en/latest/
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
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

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
    iterations_per_update=501,
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
plt.title("DynestyStatic model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()
