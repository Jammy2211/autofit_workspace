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

Load data of a 1D `Gaussian` + 1D Exponential, by loading it from a .json file in the directory 
`autofit_workspace/dataset//gaussian_x1__exponential_x1`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the data. We'll use its shape to determine the xvalues of the
data for the plot.
"""
xvalues = range(data.shape[0])
plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.show()
plt.close()

"""
__Model__

Next, we create our model, which in this case corresponds to a `Gaussian` + `Exponential`. 

The `Gaussian` has 3 parameters (centre, normalization and sigma) and Exponential 3 parameters (centre, normalization 
and rate), meaning the non-linear parameter space has dimensionality = 6.
"""
gaussian = af.Model(af.ex.Gaussian)
exponential = af.Model(af.ex.Exponential)

"""
The start-point API does not conflict with the use of priors, which are still associated with every parameter.

We manually customize the priors of the model used by the non-linear search.

We use broad `UniformPriors`'s so that our priors do not impact our inferred model and errors (which would be
the case with tight `GaussianPrior`'s.
"""
gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
We can now compose the overall model using a `Collection`, which takes the model components we defined above.
"""
model = af.Collection(gaussian=gaussian, exponential=exponential)

"""
We can inspect the model (with customized priors) via its `.info` attribute.
"""
print(model.info)

"""
__Start Point__

We now define the start point of certain parameters in the model.

By looking at the data, we can see clearly that both the `Gaussian` and `Exponential` are centred near 50.0, thus
we set both their starting points to between the range 49.0 to 51.0.

We also set the `Gaussian` `sigma` value to a start point of 4.0 to 6.0.

For all parameters where the start-point is not specified, their parameter values are drawn randomly from the prior
when determining the initial locations of the parameter.
"""
initializer = af.SpecificRangeInitializer(
    {
        model.gaussian.centre: (49.0, 51.0),
        model.gaussian.sigma: (4.0, 6.0),
        model.exponential.centre: (49.0, 51.0),
    }
)

"""
A quick look at the model's `info` attribute shows that the starting points above do not change
the priors or model info.
"""
print(model.info)

"""
__Analysis__

Create the analysis which fits the model to the data.

It fits the data as the sum of the two `Gaussian`'s in the model.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We will perform the fit using the MCMC algorithm Emcee, which samples parameter space via 50 walkers.

The starting point of all walkers will correspond to `centre` values within 49.0 to 51.0 and a `sigma` values between
4.0 to 6.0.

note that once the Emcee sampling begins, walkers are able to walk outside the starting point range set in the
initializer above.
"""
search = af.Emcee(
    path_prefix="features",
    name="start_point",
    nwalkers=50,
    nsteps=500,
    initializer=initializer,
)


result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can print the initial `parameter_lists` of the result's `Samples` object to check that the initial walker samples 
were set within the start point ranges above.
"""
samples = result.samples

print(samples.model.parameter_names)

print(samples.parameter_lists[0])
print(samples.parameter_lists[1])
print(samples.parameter_lists[2])

"""
Finish.
"""
