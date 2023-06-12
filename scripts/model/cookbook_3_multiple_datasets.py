"""
Cookbook 3: Multiple Datasets
=============================

__Prerequisites__

You should be familiar with the multiple dataset model-fitting API detailed in the following scripts and docs:

__Overview__

When fitting multiple datasets, a single identical model can be fitted to all datasets simultaneously.

Alternatively, models can be composed where certain variables vary across the datasets.

This cookbook illustrates how these different models can be composed and fitting, by interfacing the model with a
list of `Analysis` objects
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
__Identical Model Across Datasets__

We first show how to compose a model which is fit to multiple datasets, with no variation in model parameters
across the datasets. 

To perform a model-fit to multiple datasets, we interface the model with `Analysis` objects.

To make the cookbook concise, we will use `Analysis` objects without real data and not perform the actual fit,
instead simply passing `None` to the `data` and `noise_map` attributes.
"""
total_datasets = 3

analysis_list = []

for i in range(total_datasets):
    analysis = af.ex.Analysis(
        data=None, noise_map=None
    )  # real data should go here for a real model-fit.

    analysis_list.append(analysis)

"""
To fit multiple datasets simultaneously, the analysis summing API is used:
"""
analysis = sum(analysis_list)

"""
To fit all datasets simultaneously with the same model, we simply compose this model via `af.Model()`

This therefore does not require any specific model composition code.
"""


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian``s model parameters.
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


model = af.Model(af.ex.Gaussian)

"""
A fit using this model could therefore be performed as follows (had real data been used):
"""
search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_simple")

# result_list = search.fit(model=model, analysis=analysis)

"""
__Variable Model Across Datasets__

We now compose a model where the parameters vary across the datasets.

We now compose a model where: 

 - The `centre` and `normalization` values of the Gaussian fitted to every dataset in every `Analysis` object are
 identical. 
 
 - The `sigma` value of the every Gaussian fitted to every dataset in every `Analysis` object are different.

This means that the model has 5 free parameters in total, the shared `centre` and `normalization` and 5 unique
`sigma` values, one for each dataset.
"""
model = af.Collection(gaussian=af.Model(Gaussian))

analysis = sum(analysis_list)

analysis = analysis.with_free_parameters(model.gaussian.sigma)

"""
The code above does not immediately update the model. 

In fact, the model is modified immediately before a non-linear search begins (e.g. after we 
call `search.fit(model=model, analysis=analysis)`).

So that we can show how the analysis modifies the model (by printing the `model.info` attribute) we therefore call
the function below, that is called at the start of a search.

The following line of code SHOULD NOT be called in your own model-fitting script, as it will occur automatically
when the non-linear search begins.
"""
model = analysis.modify_model(model)

"""
We can now inspect how the analysis list has altered the model to add free `sigma` parameters for every dataset.
"""
print(model.prior_count)
print(model.info)

"""
We can make multiple parameters free by simply adding them to the input list above.
"""
model = af.Collection(gaussian=af.Model(Gaussian))

analysis = sum(analysis_list)

analysis = analysis.with_free_parameters(model.gaussian.sigma, model.gaussian.centre)


"""
__Variable Parameters As Relationship__

In the model above, an extra free parameter `sigma` was added for every dataset. 

This was ok for the simple model fitted here to just 3 datasets, but for more complex models and problems with 10+
datasets one will quick find that the model complexity increases dramatically.

We can compose models where the free parameter(s) vary according to a user-specified function across the datasets.

For example, we could make it so that `sigma` is computed according to:

 `y = m * x + c` : `sigma` = sigma_m * x + sigma_c`
 
Where x is an integer number specifying the index of the dataset (e.g. 1, 2 and 3).
 
By defining a relation of this form, `sigma_m` and `sigma_c` are the free parameters of the model. 
"""
model = af.Collection(gaussian=af.Model(Gaussian))

sigma_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

x_list = [1.0, 2.0, 3.0]

analysis_with_relation_list = []

for x, analysis in zip(x_list, analysis_list):
    sigma_relation = (sigma_m * x) + sigma_c

    analysis_with_relation = analysis.with_model(
        model.replacing({model.gaussian.sigma: sigma_relation})
    )

    analysis_with_relation_list.append(analysis_with_relation)

"""
To print the model `info` and see how the code above changes its composition we again have to call the `modify_model`
function first, which is automatically called before a non-linear search begins.
"""
analysis_with_relation = sum(analysis_with_relation_list)

model = analysis_with_relation.modify_model(model)

print(model.info)

"""
We can use division, subtraction and logorithms to create more complex relations and apply them to different parameters, 
for example:

 `y = m * log10(x) - log(z) + c` : `sigma` = sigma_m * log10(x) - log(z) + sigma_c` 
 `y = m * (x / z)` : `centre` = centre_m * (x / z)`
"""
model = af.Collection(gaussian=af.Model(Gaussian))

sigma_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

centre_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

x_list = [1.0, 10.0, 30.0]
z_list = [2.0, 4.0, 6.0]

analysis_with_relation_list = []

for x, z, analysis in zip(x_list, z_list, analysis_list):
    sigma_relation = (sigma_m * af.Log10(x) - af.Log(z)) + sigma_c
    centre_relation = centre_m * (x / z)

    analysis_with_relation = analysis.with_model(
        model.replacing(
            {
                model.gaussian.sigma: sigma_relation,
                model.gaussian.centre: centre_relation,
            }
        )
    )

    analysis_with_relation_list.append(analysis_with_relation)

analysis_with_relation = sum(analysis_with_relation_list)

model = analysis.modify_model(model)

print(model.info)

"""
__Example Use Case__ 

An example use-case of such a model is time-varying data, whereby each dataset is an observation as a function of time.

We may have knowledge that certain parameters do vary as a function of time, whereas others do not. We can therefore
parameterize a model which varies as a function of `time=t` of the form:

` `y = m * t + c` : `sigma` = sigma_m * t + sigma_c`
"""
