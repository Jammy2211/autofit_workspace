"""
Feature: Search Chaining
========================

To perform a model-fit, we typically compose one model and fit it to our data using one non-linear search.

Search chaining fits many different models to a dataset using a chained sequence of non-linear searches. Initial
fits are performed using simplified model parameterizations and faster non-linear fitting techniques. The results of
these simplified fits can then be used to initialize fits using a higher dimensionality model with more detailed
non-linear search.

To fit highly complex models our aim is therefore to granularize the fitting procedure into a series of **bite-sized**
searches which are faster and more reliable than fitting the more complex model straight away.

Our ability to construct chained non-linear searches that perform model fitting more accurately and efficiently relies
on our **domain specific knowledge** of the model fitting task. For example, we may know that our dataset contains
multiple features that can be accurately fitted separately before performing a joint fit, or that certain parameter
share minimal covariance such that certain parameters can be fixed before fitting both with a more complex model
parameterization.

We may also know tricks that can speed up the fitting of the initial model, for example reducing the size of the data
or changing the likelihood evaluations in a way that makes them quicker (likely at the expense of the overall
quality of the fit itself). By using chained searches these speed-ups can be relaxed towards the end of the
model-fitting sequence when we want the most precise, most accurate model that best fits the dataset available.

In this example we demonstrate search chaining using the example data where there are two `Gaussians` that are visibly
split. Instead of fitting them simultaneously using a single non-linear search consisting of N=6 parameters, we break
this model-fit into a chained of three searches where:

 1) The first model fits just the left `Gaussian` where N=3.
 2) The first model fits just the right `Gaussian` where again N=3.
 3) The final model is fitted with both `Gaussians` where N=6. Crucially, the results of the first two searches
 are used to initialize the search and tell it the highest likelihood regions of parameter space.

By initially fitting parameter spaces of reduced complexity we can achieve a more efficient and reliable model-fitting
procedure.

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
import numpy as np
from os import path

import autofit as af

"""
__Data__

Load data of two 1D Gaussians, by loading it from a .json file in the directory 
`autofit_workspace/dataset/gaussian_x1_with_feature`.

This 1D data includes two `Gaussians` that are split from one another.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2_split")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the data. 

Two separate `Gaussians` are clearly visible.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian Data With two Gaussians split apart.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Analysis__

Create the analysis which fits the model to the data.

It fits the data as the sum of as many `Gaussian`'s as are in the model.

To better fit the left gaussian, we remove all data points in the right-half of the data. Note that for more 
computationally demanding model-fitting problems this would give a significant speed-up in log likelihood function.
"""
analysis_1 = af.ex.Analysis(data=data[0:50], noise_map=noise_map[0:50])

"""
__Model__

We are now going to fit the left `Gaussian` in this split dataset. 
"""
model_1 = af.Collection(gaussian_left=af.ex.Gaussian)

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search 1__

Fit the data with the `left_gaussian` using a single non-linear search. 

Given the simplicity of the model, we can use a low number of live points to achieve a fast model-fit.
"""
search_1 = af.DynestyStatic(
    name="search[1]__left_gaussian",
    path_prefix=path.join("features", "search_chaining"),
    nlive=30,
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result 1__

The `info` attribute shows the result in a readable format.
"""
print(result_1.info)

"""
By plotting the result we can see we have fitted the left `Gaussian` reasonably well.
"""
instance = result_1.max_log_likelihood_instance

model_data = instance.gaussian_left.model_data_from(xvalues=np.arange(data.shape[0]))

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("Search 1  fit to left Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search 2 __

We now repeat the above process for the right `Gaussian`.

We could remove the data on the left like we did the `Gaussian` above. However, we are instead going to fit the full 
dataset.
"""
analysis_2 = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Model__

Because the left Gaussian is now again in the data, we need to fit it. We can do this, without increasing the 
dimensionality of our parameter space to N=6, by using the maximum log likelihood model of the `gaussian_left` in
search[1].

For search chaining, **PyAutoFit** has many convenient methods for passing the results of a search to a subsequence 
search. Below, we achieve this by passing the result of the search above as an `instance.
"""
model_2 = af.Collection(
    gaussian_left=result_1.instance.gaussian_left, gaussian_right=af.ex.Gaussian
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
We now run our second Dynesty search to fit the right `Gaussian`.

Given the simplicity of the model, we can again use a low number of live points to achieve a fast model-fit.
"""
search_2 = af.DynestyStatic(
    name="search[2]__right_gaussian",
    path_prefix=path.join("features", "search_chaining"),
    nlive=30,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result 2__

The `info` attribute shows the result in a readable format.
"""
print(result_2.info)

"""
By plotting the result we can see we have fitted the right `Gaussian` reasonably well and that the model includes the
`left_gaussian` from the first search.
"""
instance = result_2.max_log_likelihood_instance

gaussian_left = instance.gaussian_left.model_data_from(xvalues=np.arange(data.shape[0]))
gaussian_right = instance.gaussian_right.model_data_from(
    xvalues=np.arange(data.shape[0])
)

model_data = gaussian_left + gaussian_right

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_left, "--")
plt.plot(range(data.shape[0]), gaussian_right, "--")
plt.title("Search 2 fit to right Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search 3__

We now fit both `Gaussians`'s simultaneously, using the results of the previous two searches to initialize where 
the non-linear searches parameter space.

To pass the result in this way we use the command `result.model`, which in contrast to `result.instance` above passes
the parameters not as the maximum log likelihood values but as `GaussianPrior`'s that are fitted for by the
non-linear search. We discuss below how this `GaussianPrior` is passed.
"""

model_3 = af.Collection(
    gaussian_left=result_1.model.gaussian_left,
    gaussian_right=result_2.model.gaussian_right,
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1` and `result_2`.
"""
print(model_3.info)

"""
We now perform the search.
"""
analysis_3 = af.ex.Analysis(data=data, noise_map=noise_map)

search_3 = af.DynestyStatic(
    name="search[3]__both_gaussians",
    path_prefix=path.join("features", "search_chaining"),
    nlive=100,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
By plotting the result we can now see we have fitted both `Gaussian`'s accurately.
"""
instance = result_3.max_log_likelihood_instance

gaussian_left = instance.gaussian_left.model_data_from(xvalues=np.arange(data.shape[0]))
gaussian_right = instance.gaussian_right.model_data_from(
    xvalues=np.arange(data.shape[0])
)

model_data = gaussian_left + gaussian_right

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_left, "--")
plt.plot(range(data.shape[0]), gaussian_right, "--")
plt.title("Search 3 fit to both Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Prior Passing__

Now search 3 is complete, you should checkout its `model.info` file. The parameters do not use the default priors of 
the `Gaussian` model component. Instead, they use GaussianPrior`s where:

 - The mean values are the median PDF results of every parameter inferred by the fits performed in searches 1 and 2.
 - They sigma values are the errors computed by these searches, or they are values higher than these errors.

The sigma values uses the errors of searches 1 and 2 for an obvious reason, this is a reasonable estimate of where in
parameter space the model-fit can be expected to provide a good fit to the data. However, we may want to specify 
even larger sigma values on certain parameters, if for example we anticipate that our earlier searches may under 
estimate the errors.

The `width_modifier` term in the `Gaussian` section of the `config/priors/gaussian.yaml` is used instead of the errors 
of a search, when the errors estimated are smaller  than the `width_modifier` value. This ensure that the sigma 
values used in later searches do not assume extremely small values if earlier searches risk under estimating the errors.

Thus, search 3 used the results of searches 1 and 2 to inform it where to search non-linear parameter space! 

The `prior_passer` settings in the `general.yaml` config customizes how priors are passed from a search as follows:

 - sigma: The sigma value of the errors passed to set the sigma values in the previous search are estimated at.
 - use_widths: If False, the "width_modifier" values in the json_prior configs are not used to override a passed
 error value.
 - use_errors: If False, errors are not passed from search 1 to set up the priors and only the `width_modifier`
  entries in the configs are used.  

There are two ways a value is specified using the priors/width file:

 1) Absolute: In this case, the error assumed on the parameter is the value given in the config file. 
  For example, if for the width on `centre` the width modifier reads "Absolute" with a value 20.0, this means if the 
  error on the parameter `centre` was less than 20.0 in the previous search, the sigma of its `GaussianPrior` in 
  the next search will be 20.0.

 2) Relative: In this case, the error assumed on the parameter is the % of the value of the estimate value given in 
 the config file. For example, if the normalization estimated in the previous search was 2.0, and the relative error in 
 the config file reads "Relative" with a value 0.5, then the sigma of the `GaussianPrior` 
  will be 50% of this value, i.e. sigma = 0.5 * 2.0 = 1.0.

We use absolute and relative values for different parameters, depending on their properties. For example, using the 
relative value of a parameter like the `centre` makes no sense as the profile could be centred at 0.0, making 
the relative error tiny and poorly defined.

However, there are parameters where using an absolute value does not make sense. Normalization is a good example of this. 
The normalization of an image depends on its units and S/N. There is no single absolute value that one can use to 
generically chain the normalization of any two proflies. Thus, it makes more sense to chain them using the relative value 
from a previous search.

We can customize how priors are passed from the results of a search editing the `prior_passer` settings in 
the `general.yaml` config.

__EXAMPLE__

Lets go through an example using a real parameter. Lets say in search 1 we fit a `Gaussian` and we estimate that 
its normalization is equal to 4.0 +- 2.0 where the error value of 2.0 was computed at 3.0 sigma confidence. To pass this 
as a prior to search 2, we would write:

    gaussian.normalization = search_1_result.model.gaussian.normalization

The prior on the `Gaussian` `normalization` in search 2 would thus be a `GaussianPrior`, with mean=4.0 and 
sigma=2.0. If we had used a sigma value of 1.0 to compute the error, which reduced the estimate from 4.0 +- 2.0 to 
4.0 +- 1.0, the sigma of the `GaussianPrior` would instead be 1.0. 

If the error on the normalization in search 1 had been really small, lets say, 0.01, we would instead use the value of the 
normalization width in the priors config file to set sigma instead. In this case, the prior config file specifies 
that we use an "Relative" value of 0.5 to chain this prior. Thus, the GaussianPrior in search 2 would have a mean=4.0 
and sigma=2.0.

And with that, we`re done. Chaining searches is a bit of an art form, but for certain problems can be extremely 
powerful.
"""

"""
Cookbook 5: Model Linking
=========================

__Prerequisites__

You should be familiar with the search chaining API detailed in the following scripts and docs:

__Overview__

Search chaining allows one to perform back-to-back non-linear searches to fit a dataset, where the model complexity
increases after each fit.

To perform search chaining, **PyAutoFit** has tools for passing the results of one model-fit from one fit to the next,
and change its parameterization between each fit.

This cookbook is a concise reference to the model linking API.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
import os
from os import path

import autofit as af

"""
__Model-Fit__

We perform a quick model-fit, to create a `Result` object which has the attributes necessary to illustrate the model
linking API.
"""
model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

dynesty = af.DynestyStatic(name="cookbook_5_model_linking", nlive=50, sample="rwalk")

result = dynesty.fit(model=model, analysis=analysis)

"""
__Instance & Model__

The result object has two key attributes for model linking:

 - `instance`: The maximum log likelihood instance of the model-fit, where every parameter is therefore a float.

 - `model`: An attribute which represents how the result can be passed as a model-component to the next fit (the
 details of how its priors are passed are given in full below).

Below, we create a new model using both of these attributes, where:

 - All of the `gaussian` model components parameters are passed via the `instance` attribute and therefore fixed to 
 the inferred maximum log likelihood values (and are not free parameters in the model).

  - All of the `exponential` model components parameters are passed via the `model` attribute and therefore are free
  parameters in the model.

The new model therefore has 3 free parameters and 3 fixed parameters.
"""
model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

model.gaussian = result.instance.gaussian
model.exponential = result.model.exponential

"""
The `model.info` attribute shows that the parameter and prior passing has occurred as described above.
"""
print(model.info)

"""
We can print the priors of the exponenital:
"""
print("Exponential Model Priors \n")
print("centre = ", model.exponential.centre)
print("normalization = ", model.exponential.normalization)
print("rate = ", model.exponential.rate)

"""
How are the priors set via model linking? The full description is quite long, therefore it is attatched to the
bottom of this script so that we can focus on the model linking API.

__Component Specification__

Model linking can be performed on any component of a model, for example to only pass specific parameters as 
an `instance` or `model`.
"""
gaussian = af.Model(af.ex.Gaussian)

gaussian.centre = result.instance.gaussian.centre
gaussian.normalization = result.model.gaussian.normalization
gaussian.sigma = result.instance.gaussian.sigma

exponential = af.Model(af.ex.Exponential)

exponential.centre = result.model.exponential.centre
exponential.normalization = result.model.exponential.normalization
exponential.rate = result.instance.exponential.rate

model = af.Collection(gaussian=gaussian, exponential=exponential)

"""
The `model.info` attribute shows that the parameter and prior passing has occurred on individual components.
"""
print(model.info)

"""
__Take Attributes__

The examples above linked models where the individual model components that were passed stayed the same.

We can link two related models, where only a subset of parameters are shared, by using the `take_attributes()` method. 

For example, lets define a `GaussianKurtosis` which is a `Gaussian` with an extra parameter for its kurtosis.
"""


class GaussianKurtosis:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian``s model parameters.
        sigma=5.0,
        kurtosis=1.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma
        self.kurtosis = kurtosis


"""
The `take_attributes()` method takes a `source` model component, and inspects the names of all its parameters. 

For  the `Gaussian` model result input below, it finds the parameters `centre`, `normalization` and `sigma`.

It then finds all parameters in the new `model` which have the same names, which for the `GaussianKurtosis` is
`centre`, `normalization` and `sigma`.

For all parameters which have the same name, the parameter is passed. 
"""
model = af.Collection(gaussian=af.Model(GaussianKurtosis))
model.kurtosis = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)

model.gaussian.take_attributes(source=result.model.gaussian)

"""
Because the result was passed using `model` we see the priors on the `GaussianKurtosis` `centre`, 
`normalization` and `sigma` have been updated, whereas its `kurtosis` has not.
"""
print("GaussianKurtosis Model Priors After Take Attributes via Model \n")
print("centre = ", model.gaussian.centre)
print("normalization = ", model.gaussian.normalization)
print("sigma = ", model.gaussian.sigma)
print("kurtosis = ", model.gaussian.kurtosis)

"""
If we pass `result.instance` to take_attributes the same name linking is used, however parameters are passed as
floats.
"""
model = af.Collection(gaussian=af.Model(GaussianKurtosis))
model.kurtosis = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)

model.gaussian.take_attributes(source=result.instance.gaussian)

print("Gaussian Model Priors After Take Attributes via Instance \n")
print("centre = ", model.gaussian.centre)
print("normalization = ", model.gaussian.normalization)
print("sigma = ", model.gaussian.sigma)
print("kurtosis = ", model.gaussian.kurtosis)

"""
__As Model__

A common problem is when we have an `instance` (e.g. from a previous fit where we fixed the parameters)
but now wish to make its parameters free parameters again.

Furthermore, we may want to do this for specific model components.

The `as_model` method allows us to do this. Below, we pass the entire result (e.g. both the `gaussian` 
and `exponential` components), however we pass the `Gaussian` class to `as_model`, meaning that any model
component in the `instance` which is a `Gaussian` will be converted to a model with free parameters.
"""
model = result.instance.as_model((af.ex.Gaussian,))

print("Gaussian Model Priors After via as_model: \n")
print("centre = ", model.gaussian.centre)
print("normalization = ", model.gaussian.normalization)
print("sigma = ", model.gaussian.sigma)
print("centre = ", model.exponential.centre)
print("normalization = ", model.exponential.normalization)
print("rate= ", model.exponential.rate)

"""
The `as_model()` method does not have too much utility for the simple model used in this cookbook. 

However, for multi-level models with many components, it is a powerful tool to compose custom models.
"""


class MultiLevelProfiles:
    def __init__(
        self,
        higher_level_centre=50.0,  # This is the centre of all Gaussians in this multi level component.
        profile_list=None,  # This will contain a list of model-components
    ):
        self.higher_level_centre = higher_level_centre

        self.profile_list = profile_list


group_level_0 = af.Model(
    MultiLevelProfiles, profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Gaussian]
)

group_level_1 = af.Model(
    MultiLevelProfiles,
    profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Exponential],
)

model = af.Collection(group_level_0=group_level_0, group_level_1=group_level_1)

"""
This means every `Gaussian` in the complex multi-level model above would  have parameters set via the result of our
model-fit, if the model above was fitted such that it was contained in the result.
"""
model = result.instance.as_model((af.ex.Gaussian,))
