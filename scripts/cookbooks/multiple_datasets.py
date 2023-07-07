"""
Feature: Multiple Datasets
==========================

It is common to have multiple observations of the same signal. For the 1D Gaussian example, this would be multiple
1D datasets of the same underlying Gaussian, with different noise-map realizations. In this situation, fitting the
same model to all datasets simultaneously is desired, and would provide better constraints on the model.

On other occations, the signal may vary across the datasets in a way that requires that the model is updated
accordingly. For example, a scenario where the centre of each Gaussian is the same across the datasets, but
their `sigma` values are different in each dataset. A model where all Gaussians share the same `centre` is now required.

This examples illustrates how to perform model-fits to multiple datasets simultaneously, including tools to customize
the model composition such that specific parameters of the model vary across the datasets.

This uses the summing of `Analysis` object, which each have their own unique dataset and `log_likelihood_function`.
Unique `Analysis` objects can be written for each dataset, meaning that we can perform model-fits to diverse datasets
with different formats and structures.

It is also common for each individual dataset to only constrain specific aspects of a model. The high level of model
customizaiton ensures that composing a model that is appropriate for fitting to such large datasets is straight
forward.

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
import autofit.plot as aplt

"""
__Data__

First, lets load 3 datasets of a 1D Gaussian, by loading them from .json files in the directory 
`autofit_workspace/dataset/`.

All three datasets contain an identical signal, meaning that it is appropriate to fit the same model to all three 
datasets simultaneously.

Each dataset has a different noise realization, meaning that performing a simultaneously fit will offer improved 
constraints over individual fits.
"""
dataset_size = 3

data_list = []
noise_map_list = []

for dataset_index in range(dataset_size):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1_identical_{dataset_index}"
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    data_list.append(data)

    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )
    noise_map_list.append(noise_map)

"""
Now lets plot all 3 datasets, including their error bars. 
"""
for data, noise_map in zip(data_list, noise_map_list):
    xvalues = range(data.shape[0])

    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        linestyle=" ",
        elinewidth=1,
        capsize=2,
    )
    plt.show()
    plt.close()

"""
__Model__

Next, we create our model, which corresponds to a single 1D Gaussian, that is fitted to all 3 datasets simultaneously.
"""
model = af.Model(af.ex.Gaussian)

"""
Checkout `autofit_workspace/config/priors/model.yaml`, this config file defines the default priors of the `Gaussian` 
model component. 

We overwrite the priors below to make them explicit.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We set up our three instances of the `Analysis` class, using the class described in `analysis.py`.

We set up an `Analysis` for each dataset one-by-one, using a for loop:
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    analysis_list.append(analysis)

"""
__Analysis Summing__

We now sum together every analysis in the list, to produce an overall analysis class which we fit with the non-linear
search.

By summing analysis objects the following happen:

 - The log likelihood values computed by the `log_likelihood_function` of each individual analysis class are summed to
   give an overall log likelihood value that the non-linear search uses for model-fitting.

 - The output path structure of the results goes to a single folder, which includes sub-folders for the visualization
   of every individual analysis object based on the `Analysis` object's `visualize` method.
"""
analysis = analysis_list[0] + analysis_list[1] + analysis_list[2]

"""
We can alternatively sum a list of analysis objects as follows:
"""
analysis = sum(analysis_list)

"""
The `log_likelihood_function`'s can be called in parallel over multiple cores by changing the `n_cores` parameter:
"""
analysis.n_cores = 1

"""
__Search__

To fit multiple datasets via a non-linear search we use this summed analysis object:
"""
search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_simple")

result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit is a list of the `Result` objects you have used in other examples.

In this example, the same model is fitted across all analyses, thus every `Result` in the `result_list` contains
the same information on the samples and thus gives the same output from methods such 
as `max_log_likelihood_instance`.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
We can plot the model-fit to each dataset by iterating over the results:
"""
for result in result_list:
    instance = result.max_log_likelihood_instance

    model_data = instance.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )

    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.plot(xvalues, model_data, color="r")
    plt.title("Dynesty model fit to 1D Gaussian dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

"""
__Variable Model Across Datasets__

Above, the same model was fitted to every dataset simultaneously, which was possible because all 3 datasets contained 
an identical signal with only the noise varying across the datasets.

It is common for the signal in each dataset to be different and for it to constrain only certain aspects of the model.
The model parameterization therefore needs to change in order to account for this.

Lets look at an example of a dataset of 3 1D Gaussians where the signal varies across the datasets:
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_variable")

dataset_name_list = ["sigma_0", "sigma_1", "sigma_2"]

data_list = []
noise_map_list = []

for dataset_name in dataset_name_list:
    dataset_time_path = path.join(dataset_path, dataset_name)

    data = af.util.numpy_array_from_json(
        file_path=path.join(dataset_time_path, "data.json")
    )
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_time_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)

"""
If we plot these datasets, we see that the `sigma` of each Gaussian decreases.

We will illustrate models which vary over the data based on this `sigma` value. 
"""
for data, noise_map in zip(data_list, noise_map_list):
    xvalues = range(data.shape[0])

    af.ex.plot_profile_1d(xvalues=xvalues, profile_1d=data)

"""
In this case, the `centre` and `normalization` of all three 1D Gaussians are the same in each dataset,
but their `sigma` values are decreasing.

We therefore wish to compose and to fit a model to all three datasets simultaneously, where the `centre` 
and `normalization` are the same across all three datasets but the `sigma` value is unique for each dataset.

To do that, we interface a model with a summed list of analysis objects, which we create below for this new
dataset:
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    analysis_list.append(analysis)

analysis = sum(analysis_list)

"""
We next compose a model of a 1D Gaussian, as performed frequently throughout the **PyAutoFit** examples:
"""
model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

"""
We now update the model using the summed `Analysis `objects to compose a model where: 

 - The `centre` and `normalization` values of the Gaussian fitted to every dataset in every `Analysis` object are
 identical. 

 - The `sigma` value of the every Gaussian fitted to every dataset in every `Analysis` object are different.

This means that the model has 5 free parameters in total, the shared `centre` and `normalization` and a unique
`sigma` value for every dataset.
"""
analysis = analysis.with_free_parameters(model.gaussian.sigma)

"""
To inspect this new model, with extra parameters for each dataset created, we have to extract the modified version of this model from the Analysis object.
This occurs automatically when we begin a non-linear search, therefore the normal model we created above is what we input to the search.fit() method.
"""

model_updated = analysis.modify_model(model)

"""
This means that the model has 5 free parameters in total, the shared `centre` and `normalization` and a unique
`sigma` value for every dataset.
"""
print(model_updated.total_free_parameters)

"""
We can now fit this model to the data using the usual **PyAutoFit** tools:
"""
search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_free_sigma")

result_list = search.fit(model=model, analysis=analysis)

"""
__Individual Sequential Searches__

The API above is used to create a model with free parameters across ``Analysis`` objects, which are all fit
simultaneously using a summed ``log_likelihood_function`` and single non-linear search.

Each ``Analysis`` can be fitted one-by-one, using a series of multiple non-linear searches, using
the ``fit_sequential`` method.
"""
search = af.DynestyStatic(
    path_prefix="features", name="multiple_datasets_free_sigma__sequential"
)

result_list = search.fit_sequential(model=model, analysis=analysis)

"""
The benefit of this method is for complex high dimensionality models (e.g. when many parameters are passed
to `` analysis.with_free_parameters``, it breaks the fit down into a series of lower dimensionality non-linear
searches that may convergence on a solution more reliably.

__Variable Parameters As Relationship__

In the model above, an extra free parameter `sigma` was added for every dataset. 

This was ok for the simple model fitted here to just 3 datasets, but for more complex models and problems with 10+
datasets one will quick find that the model complexity increases dramatically.

In these circumstances, one can instead compose a model where the parameters vary smoothly across the datasets
via a user defined relation.

Below, we compose a model where the `sigma` value fitted to each dataset is computed according to:

 `y = m * x + c` : `sigma` = sigma_m * x + sigma_c`

Where x is an integer number specifying the index of the dataset (e.g. 1, 2 and 3).

By defining a relation of this form, `sigma_m` and `sigma_c` are the only free parameters of the model which vary
across the datasets. 

Therefore, if more datasets are added the number of model parameter does not increase, like we saw above.
"""
sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
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
We can fit this model as per usual, you may wish to checkout the `model.info` file to see how a schematic of this
model's composition.
"""
analysis_with_relation = sum(analysis_with_relation_list)

search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_relation")

result_list = search.fit(model=model, analysis=analysis_with_relation)

"""
__Temporally Varying Models__

An obvious example of fitting models which vary across datasets are time-varying models, where the datasets are
observations of a signal which varies across time.

In such circumstances, it is common for certain model parameters to be known to not vary as a function of time (and 
therefore be fixed across the datasets) whereas other parameters are known to vary as a function of time (and therefore
should be parameterized accordingly using the API illustrated here).

__Different Analysis Objects__

For simplicity, this example summed together only a single `Analysis` class. 

For many problems one may have multiple datasets which are quite different in their format and structure (perhaps 
one is a  1D signal whereas another dataset is an image). In this situation, one can simply define unique `Analysis`
objects for each type of dataset, which will contain a unique `log_likelihood_function` and methods for visualization.

Nevertheless, the analysis summing API illustrated here will still work, meaning that **PyAutoFit** makes it simple to 
fit highly customized models to multiple datasets that are different in their format and structure. 

__Graphical Models__

A common class of models used for fitting complex models to large datasets are graphical models. 

Graphical models can include addition parameters not specific to individual datasets describing the overall 
relationship between different model components, thus allowing one to infer the global trends contained within a 
dataset.

**PyAutoFit** has a dedicated feature set for fitting graphical models and interested readers should
checkout the graphical modeling chapter of **HowToFit** (https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_graphical_models.html)

__Wrap Up__

We have shown how **PyAutoFit** can fit large datasets simultaneously, using custom models that vary specific
parameters across the dataset.

The `autofit_workspace/*/model/cookbook_3_multiple_datasets` cookbook gives a concise API reference for model 
composition when fitting multiple datasets.
"""
