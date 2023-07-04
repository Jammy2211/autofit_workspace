"""
__Example: Result__

In this example, we'll repeat the fit performed in `fit.py` of 1D data of a `Gaussian` + Exponential profile with 1D line
data using the  non-linear  search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files `example/model.py`,`example/analysis.py` and `example/fit.py` to
see how the fit is performed by the code below. The first section of code below is simply repeating the commands in
`example/fit.py`, so feel free to skip over it until you his the `Result`'s section.

The attributes of the Result object are described in `overview/simple/result.py`. This example will not cover the
attributes in full, and instead only focus on how the use of a more complex model changes the Result object.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt

from os import path
import matplotlib.pyplot as plt
import numpy as np

"""
__Data__

Load data of a 1D `Gaussian` + 1D Exponential, by loading it from a .json file in the directory 
`autofit_workspace/dataset/`, which simulates the noisy data we fit (check it out to see how we simulate the 
data).
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model__

Next, we create our model, which in this case corresponds to a `Gaussian` + Exponential.
 
We use the `af.ex` package to create the `Gaussian` and `Exponential`, where `af.ex` stands for **PyAutoFit** 
example objects used in example scripts such as this one. 

The `Gaussian` and `Exponential` models below are identical to the 1D profiles used in the `fit.py` example script.
"""
model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

"""
Checkout `autofit_workspace/config/priors` - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
__Analysis__

We now set up our `Analysis`, which describes how given an instance of our model (a Gaussian) we fit the data and 
return a log likelihood value. 

We use the `af.ex` package to create the `Analysis`, where `af.ex` stands for **PyAutoFit** example objects used
in example scripts such as this one. 

The `Analysis` below is an identical to the one used in the `fit.py` example script.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
We now create the `Emcee` non-linear search and perform a model-fit to get the result.
"""
search = af.Emcee(
    nwalkers=50,
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

Here, we'll look in detail at what information is contained in the `Result`.

It contains an `info` attribute which prints the result in readable format.
"""
print(result.info)

"""
__Parameters__

First, we can note that the parameters list of lists now has 6 entries in the parameters column, given the 
dimensionality of the model has increased from N=3 to N=6.
"""
samples = result.samples
print("All Parameters:")
print(samples.parameter_lists)
print("Sample 10`s sixth parameter value (Exponential -> rate)")
print(samples.parameter_lists[9][5], "\n")

"""
__Instances__

When we return a result as an instance, it provides us with instances of the model using the Python classes used to
compose it. 

Because our fit uses a Collection (as opposed to a `Model` in the simple example) the instance
returned a dictionary named acoording to the names given to the Collection, which above were `gaussian` and
`exponential`.

For complex models, with a large number of model components and parameters, this offers a readable API to interpret
the results.
"""
max_lh_instance = samples.max_log_likelihood()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", max_lh_instance.gaussian.centre)
print("Normalization = ", max_lh_instance.gaussian.normalization)
print("Sigma = ", max_lh_instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", max_lh_instance.exponential.centre)
print("Normalization = ", max_lh_instance.exponential.normalization)
print("Sigma = ", max_lh_instance.exponential.rate, "\n")

"""
__Vectors__

1D vectors containing models have the same meaning as before, but they are also now of size 6 given the increase in
model complexity.
"""
print("Result and Error Vectors:")
print(samples.median_pdf(as_instance=False))
print(samples.max_log_likelihood(as_instance=False))
print(samples.max_log_posterior(as_instance=False))
print(samples.values_at_upper_sigma(sigma=3.0, as_instance=False))
print(samples.values_at_lower_sigma(sigma=3.0, as_instance=False))
print(samples.errors_at_upper_sigma(sigma=3.0, as_instance=False))
print(samples.errors_at_lower_sigma(sigma=3.0, as_instance=False), "\n")

"""
__Labels__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:
 
 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
__Plot__

Because results are returned as instances, it is straight forward to use them and their associated functionality
to make plots of the results:
"""
model_gaussian = max_lh_instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = max_lh_instance.exponential.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D `Gaussian` + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Samples Filtering (Advanced)__

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.with_paths([("gaussian", "centre")])

print("All parameters of the very first sample (containing only the Gaussian centre.")
print(samples.parameter_lists[0])

print("Maximum Log Likelihood Model Instances (containing only the Gaussian centre):\n")
print(samples.max_log_likelihood(as_instance=False))

"""
Above, we specified each path as a list of tuples of strings. 

This is how the PyAutoFit source code stores the path to different components of the model, but it is not 
in-profile_1d with the PyAutoFIT API used to compose a model.

We can alternatively use the following API:
"""
samples = result.samples

samples = samples.with_paths(["gaussian.centre"])

print("All parameters of the very first sample (containing only the Gaussian centre).")
print(samples.parameter_lists[0])

"""
Above, we filtered the `Samples` but asking for all parameters which included the path ("gaussian", "centre").

We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the Gaussian's `centre` to be left with 2 parameters; the `normalization` and `sigma`.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.without_paths(["gaussian.centre"])

print(
    "All parameters of the very first sample (containing only the Gaussian normalization and sigma)."
)
print(samples.parameter_lists[0])

"""
__Wrap Up__

Adding model complexity does not change the behaviour of the Result object, other than the switch
to Collections meaning that our instances now have named entries.

When you name your model components, you should make sure to give them descriptive and information names that make 
the use of a result object clear and intuitive!
"""
