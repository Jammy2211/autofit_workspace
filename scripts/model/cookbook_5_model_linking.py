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


multi_level_0 = af.Model(
    MultiLevelProfiles, profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Gaussian]
)

multi_level_1 = af.Model(
    MultiLevelProfiles,
    profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Exponential],
)

model = af.Collection(multi_level_0=multi_level_0, multi_level_1=multi_level_1)

"""
This means every `Gaussian` in the complex multi-level model above would  have parameters set via the result of our
model-fit, if the model above was fitted such that it was contained in the result.
"""
model = result.instance.as_model((af.ex.Gaussian,))

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

The `PriorPasser` customizes how priors are passed from a search as follows:

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

We can customize how priors are passed from the results of a search and `NonLinearSearch` by inputting to the search 
a `PriorPasser` object:
"""
search = af.DynestyStatic(
    prior_passer=af.PriorPasser(sigma=2.0, use_widths=False, use_errors=True)
)

"""
The `PriorPasser` allows us to customize at what sigma the error values the model results are computed at to compute
the passed sigma values and customizes whether the widths in the config file, these computed errors, or both, 
are used to set the sigma values of the passed priors.

The default values of the `PriorPasser` are found in the config file of every non-linear search, in the [prior_passer]
section. All non-linear searches by default use a sigma value of 3.0, use_width=True and use_errors=True.

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
