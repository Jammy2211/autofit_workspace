"""
Feature: Start Point
====================

For maximum likelihood estimator (MLE) and Markov Chain Monte Carlo (MCMC) non-linear searches, parameter space
sampling is built around having a "location" in parameter space.

This could simply be the parameters of the current maximum likelihood model in an MLE fit, or the locations of many
walkers in parameter space (e.g. MCMC).

For many model-fitting problems, we may have an expectation of where correct solutions lie in parameter space and
therefore want our non-linear search to start near that location of parameter space. Alternatively, we may want to
sample a specific region of parameter space, to determine what solutions look like there.

The start-point API allows us to do this, by manually specifying the start-point of an MLE fit or the start-point of
the walkers in an MCMC fit. Because nested sampling draws from priors, it cannot use the start-point API.

__Comparison to Priors__

Similar behaviour can be achieved by customizing the priors of a model-fit. We could place `GaussianPrior`'s
centred on the regions of parameter space we want to sample, or we could place tight `UniformPrior`'s on regions
of parameter space we believe the correct answer lies.

The downside of using priors is that our priors have a direct influence on the parameters we infer and the size
of the inferred parameter errors. By using priors to control the location of our model-fit, we therefore risk
inferring a non-representative model.

For users more familiar with statistical inference, adjusting ones priors in the way described above leads to
changes in the posterior, which therefore impacts the model inferred.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects you have seen elsewhere in the workspace.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
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
__Start Point Priors__

The start-point API does not conflict with the use of priors, which are still associated with every parameter.

We manually customize the priors of the model used by the non-linear search.

We use broad `UniformPriors`'s so that our priors do not impact our inferred model and errors (which would be
the case with tight `GaussianPrior`'s.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

"""
We can inspect the model (with customized priors) via its `.info` attribute.
"""
print(model.info)


"""
__Start Point__

We now define the start point of certain parameters in the model:

 - The 1D Gaussian is centred near pixel 50, so we set a start point there.

 - The sigma value of the Gaussian looks around 10, so we set a start point there.

For all parameters where the start-point is not specified (in this case the `normalization`, their 
parameter values are drawn randomly from the prior when determining the initial locations of the parameters.
"""
initializer = af.InitializerParamBounds(
    {model.centre: (49.0, 51.0), model.sigma: (9.0, 11.0)}
)

"""
A quick look at the model's `info` attribute shows that the starting points above do not change
the priors or model info.
"""
print(model.info)

"""
Information on the initializer can be extracted and printed, which is shown below, where the start points are
clearly visible.
"""
print(initializer.info_from_model(model=model))


"""
__Search + Analysis + Model-Fit__

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
search = af.Emcee(
    path_prefix="searches",
    name="start_point",
    nwalkers=30,
    nsteps=1000,
    initializer=initializer,
    number_of_cores=1,
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can print the initial `parameter_lists` of the result's `Samples` object to check that the initial 
walker samples were set within the start point ranges above.
"""
samples = result.samples

print(samples.model.parameter_names)

print(samples.parameter_lists[0])
print(samples.parameter_lists[1])
print(samples.parameter_lists[2])

"""
Finish.
"""
