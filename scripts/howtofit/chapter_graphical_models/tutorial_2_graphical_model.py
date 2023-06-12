"""
Tutorial 2: Graphical Models
============================

In the previous tutorial, we fitted a dataset containing 5 noisy 1D Gaussian which had a shared `centre` value. We
attempted to estimate the `centre` by fitting each dataset individually and combining the value of the `centre`
inferred by each fit into an overall estimate, using a weighted average.

Graphical modeling follows a different approach. It composes a single model that is fitted to the entire dataset.
This model includes specific model component for every individual 1D Gaussian in the sample. However, the graphical
model also has shared parameters between these individual model components.

This example fits a graphical model using the same sample fitted in the previous tutorial, consisting of
data of three 1D Gaussians. We fit the `Gaussian` model to each 1D gaussian. However, whereas previously
the `cenyre` of each model component was a free parameter in each fit, in the graphical model there is only
a single value of `centre` shared by all three 1D Gaussians (which is how the galaxy data was simulated).

This graphical model creates a non-linear parameter space that has parameters for every Gaussian in our sample. In this
example, there are 5 Gaussians each with their own model, therefore:

 - Each Gaussian has 2 free parameters from the components that are not shared (`normalization`, `sigma`).
 - There is one additional free parameter, which is the `centre` shared by all 5 Gaussians.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and 
 `visualize` functions.
 
 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you have seen 
and used elsewhere throughout the workspace.

__Dataset__

For each dataset we now set up the correct path and load it. 

Whereas in the previous tutorial we fitted each dataset one-by-one, in this tutorial we instead store each dataset 
in a list so that we can set up a single model-fit that fits the 5 datasets simultaneously.
"""
total_datasets = 5

dataset_name_list = []
data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
By plotting the Gaussians we can remind ourselves that determining their centres by eye is difficult.
"""
for dataset_name, data in zip(dataset_name_list, data_list):
    af.ex.plot_profile_1d(
        xvalues=np.arange(data.shape[0]),
        profile_1d=data,
        title=dataset_name,
        ylabel="Data Values",
        color="k",
    )

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class. 
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__

We now compose the graphical model that we fit, using the `Model` object you are now familiar with.

We begin by setting up a shared prior for `centre`. 

We set up this up as a single `GaussianPrior` which is passed to separate `Model`'s for each `Gaussian` below.
"""
centre_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

"""
We now set up a list of `Model`'s, each of which contain a `Gaussian` that is used to fit each of the datasets 
loaded above.

All of these `Model`'s use the `centre_shared_prior`. This means all model-components use the same value of `centre` 
for every model composed and fitted by the `NonLinearSearch`. 

For a fit using five Gaussians, this reduces the dimensionality of parameter space from N=15 (e.g. 3 parameters per 
Gaussian) to N=11 (e.g. 5 `sigma`'s 5 `normalizations` and 1 `centre`).
"""
model_list = []

for model_index in range(len(data_list)):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
    gaussian.normalization = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

    model_list.append(gaussian)

"""
__Analysis Factors__

Above, we composed a model consisting of three `Gaussian`'s with a shared `centre` prior. We also loaded three datasets
which we intend to fit with each of these `Gaussians`, setting up each in an `Analysis` class that defines how the 
model is used to fit the data.

We now simply pair each model-component to each `Analysis` class, so that **PyAutoFit** knows that: 

- `gaussian_0` fits `data_0` via `analysis_0`.
- `gaussian_1` fits `data_1` via `analysis_1`.
- `gaussian_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called an `AnalysisFactor`. 

This term is used to denote that we are composing a graphical model, which is commonly termed a 'factor graph'. A 
factor defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between 
these different nodes then define the global model we are fitting.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We combine our `AnalysisFactor`'s to compose a factor graph.

What is a factor graph? A factor graph defines the graphical model we have composed. For example, it defines the 
different model components that make up our model (e.g. the three `Gaussian` classes) and how their parameters are 
linked or shared (e.g. that each `Gaussian` has its own unique `normalization` and `sigma`, but a shared `centre` 
parameter).

This is what our factor graph looks like (visualization of graphs not implemented in **PyAutoFit** yet): 

The factor graph above is made up of two components:

- Nodes: these are points on the graph where we have a unique set of data and a model that is made up of a subset of 
our overall graphical model. This is effectively the `AnalysisFactor` objects we created above. 

- Links: these define the model components and parameters that are shared across different nodes and thus retain the 
same values when fitting different datasets.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
The fit will use the factor graph's `global_prior_model`, which uses the models contained in every analysis factor 
to contrast the overall global model that is fitted.

Printing the `info` attribute of this model reveals the overall structure of the model, which is grouped in terms
of the analysis factors and therefore datasets.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and use it to the fit the factor graph, using its `global_prior_model` property.
"""
search = af.DynestyStatic(
    path_prefix=path.join("howtofit", "chapter_graphical_models"),
    name="tutorial_2_graphical_model",
    sample="rwalk",
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same struture of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
We can now inspect the inferred value of `centre`, and compare this to the value we estimated in the previous tutorial
via a weighted average or posterior multiplicaition using KDE.(feature missing currently). 

(The errors of the weighted average and KDE below is what was estimated for a run on my PC, yours may be slightly 
different!)
"""
print(
    f"Weighted Average Centre Estimate = 48.535531422571886 (4.139907734505303) [1.0 sigma confidence intervals] \n"
)

centre = result.samples.median_pdf()[0].centre

u1_error = result.samples.values_at_upper_sigma(sigma=1.0)[0].centre
l1_error = result.samples.values_at_lower_sigma(sigma=1.0)[0].centre

u3_error = result.samples.values_at_upper_sigma(sigma=3.0)[0].centre
l3_error = result.samples.values_at_lower_sigma(sigma=3.0)[0].centre

print("Inferred value of the shared centre via a graphical model fit: \n")
print(f"{centre} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{centre} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

"""
The graphical model's centre estimate and errors are pretty much exactly the same as the weighted average or KDE!

Whats the point of fitting a graphical model if the much simpler approach of the previous tutorial gives the
same answer? 

The answer, is model complexity. Graphical models become more powerful as we make our model more complex,
our non-linear parameter space higher dimensionality and the degeneracies between different parameters on the graph
more significant. 

We will demonstrate this in the next tutorial.

__Wrap Up__

In this tutorial, we showed that for our extremely simple model the graphical model gives pretty much the
same estimate of the 1D Gaussian centre's as simpler approaches followed in the previous tutorial. 

We will next show the strengths of graphical models by fitting more complex models.
"""
