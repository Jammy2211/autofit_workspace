"""
Tutorial 4: Hierarchical
========================

In the previous tutorial, we fitted a graphical model with the aim of determining an estimate of shared parameters,
the `centre`'s of a dataset of 1D Gaussians. We did this by fitting all datasets simultaneously. When there are shared
parameters in a model, this is a powerful and effective tool, but things may not always be so simple.

A common extension to the problem is one where we expect that the shared parameter(s) of the model do not have exactly
the same value in every dataset. Instead, our expectation is that the parameter(s) are drawn from a common
parent distribution (e.g. a Gaussian distribution). It is the parameters of this distribution that we consider shared
across the dataset (e.g. the means and scatter of the Gaussian distribution). These are the parameters we ultimately
wish to infer to understand the global behaviour of our model.

This is called a hierarchical model, which we fit in this tutorial. The `centre` of each 1D Gaussian is now no
longer the same in each dataset and they are instead drawn from a shared parent Gaussian distribution
(with `mean=50.0` and `sigma=10.0`). The hierarchical model will recover the `mean` and `sigma` values of the parent
distribution'.
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

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you 
have seen and used elsewhere throughout the workspace.

__Dataset__

For each dataset we now set up the correct path and load it. 

We are loading a new Gaussian dataset, where the Gaussians have different centres which were drawn from a parent
Gaussian distribution with a mean centre value of 50.0 and sigma of 10.0.
"""
total_datasets = 5

dataset_name_list = []
data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__hierarchical", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
By plotting the Gaussians we can just about make out that their centres are not all at pixel 50, and are spread out
around it (albeit its difficult to be sure, due to the low signal-to-noise of the data). 
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

For each dataset we now create a corresponding `Analysis` class, like in the previous tutorial.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)


"""
__Model Individual Factors__

We first set up a model for each `Gaussian` which is individually fitted to each 1D dataset, which forms the
factors on the factor graph we compose. 

This uses a nearly identical for loop to the previous tutorials, however a shared `centre` is no longer used and each 
`Gaussian` is given its own prior for the `centre`. 

We will see next how this `centre` is used to construct the hierarchical model.
"""
model_list = []

for model_index in range(len(data_list)):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = af.GaussianPrior(
        mean=50.0, sigma=20.0, lower_limit=0.0, upper_limit=100.0
    )
    gaussian.normalization = af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.0)
    gaussian.sigma = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)

    model_list.append(gaussian)

"""
__Analysis Factors__

Now we have our `Analysis` classes and model components, we can compose our `AnalysisFactor`'s.

These are composed in the same way as for the graphical model in the previous tutorial.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)


"""
__Model__

We now compose the hierarchical model that we fit, using the individual Gaussian model components created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `centre` of each individual `Gaussian` dataset is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""

hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.GaussianPrior(mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0),
    sigma=af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0),
)

"""
We now add each of the individual model `Gaussian`'s `centre` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `centre` of every `Gaussian` in our dataset is now assumed 
to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping to 
estimate.
"""

for model in model_list:
    hierarchical_factor.add_drawn_variable(model.centre)

"""
__Factor Graph__

We now create the factor graph for this model, using the list of `AnalysisFactor`'s and the hierarchical factor.

Note that the `hierarchical_factor` is passed in below, which was not the case in previous tutorials.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
The factor graph model `info` attribute shows that the hierarchical factor's parameters are included in the model.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and used it to the fit the hierarchical model, again using 
its `global_prior_model` property.
"""
search = af.DynestyStatic(
    path_prefix=path.join("howtofit", "chapter_graphical_models"),
    name="tutorial_4_hierarchical",
    sample="rwalk",
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows the result, including the hierarchical factor's parameters.
"""
print(result.info)

"""
We can now inspect the inferred value of hierarchical factor's mean and sigma.

We see that they are consistent with the input values of `mean=50.0` and `sigma=10.0`.

The hierarchical factor results are at the end of the samples list, hence why we extract them using `[-1]` and [-2]`
below.
"""
samples = result.samples

mean = samples.median_pdf(as_instance=False)[-2]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-2]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-2]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-2]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-2]

print(
    "Inferred value of the mean of the parent hierarchical distribution for the centre: \n"
)
print(f"{mean} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{mean} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

scatter = samples.median_pdf(as_instance=False)[-1]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-1]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-1]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-1]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-1]

print(
    "Inferred value of the scatter (the sigma value of the Gassuain) of the parent hierarchical distribution for the centre: \n"
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

"""
__Comparison to One-by-One Fits__

We can compare the inferred values above to the values inferred for individual fits in the 
tutorial `tutorial_optional_hierarchical_individual.py`.

This fits the hierarchical model is a much simpler way -- fitting each dataset one-by-one and then fitting the 
parent Gaussian distribution to those results.

For the results below, inferred on my laptop, we can see that the correct mean and scatter of the parent Gaussian is 
inferred but the errors are much larger than the graphical model fit above.
"""
print(
    "Inferred value of the mean of the parent hierarchical distribution for one-by-one fits: \n"
)
print(
    "50.00519854538594 (35.825675441265815 65.56274024242403) [1.0 sigma confidence intervals]"
)
print(
    "50.00519854538594 (1.3226539914914734 96.92151898283811) [3.0 sigma confidence intervals]"
)

print(
    "Inferred value of the scatter of the parent hierarchical distribution for one-by-one fits: \n"
)
print(
    "15.094393493747617 (4.608862348173649 31.346751522582483) [1.0 sigma confidence intervals]"
)
print(
    "15.094393493747617 (0.060533647989089806 49.05537884440667) [3.0 sigma confidence intervals]"
)

"""
__Benefits of Graphical Model__

We compared the results inferred in this script via a graphical model to a simpler approach which fits each dataset 
one-by-one and infers the hierarchical parent distribution's parameters afterwards.

The graphical model provides a more accurate and precise estimate of the parent distribution's parameters. This is 
because the fit to each dataset informs the hierarchical distribution's parameters, which in turn improves
constraints on the other datasets. In a hierarchical fit, we describe this as "the datasets talking to one another". 

For example, by itself, dataset_0 may give weak constraints on the centre spanning the range 20 -> 85 at 1 sigma 
confidence. Now, consider if simultaneously all of the other datasets provide strong constraints on the 
hierarchical's distribution's parameters, such that its `mean = 50 +- 5.0` and `sigma = 10.0 +- 2.0` at 1 sigma 
confidence. 

This will significantly change our inferred parameters for dataset 0, as the other datasets inform us
that solutions where the centre is well below approximately 40 are less likely, because they are inconsistent with
the parent hierarchical distribution's parameters!

For complex graphical models with many hierarchical factors, this phenomena of the "datasets talking to one another" 
is crucial in breaking degeneracies between parameters and maximally extracting information from extremely large
datasets.

__Wrap Up__

By composing and fitting hierarchical models in the graphical modeling framework we can fit for global trends
within large datasets. The tools applied in this tutorial and the previous tutorial can be easily extended to 
compose complex graphical models, with multiple shared parameters and hierarchical factors.

However, there is a clear challenge scaling the graphical modeling framework up in this way: model complexity. As the 
model becomes more complex, an inadequate sampling of parameter space will lead one to infer local maxima. Furthermore,
one will soon hit computational limits on how many datasets can feasibly be fitted simultaneously, both in terms of
CPU time and memory limitations. 

Therefore, the next tutorial introduces expectation propagation, a framework that inspects the factor graph of a 
graphical model and partitions the model-fit into many separate fits on each graph node. When a fit is complete, 
it passes the information learned about the model to neighboring nodes. 

Therefore, graphs comprising hundreds of model components (and tens of thousands of parameters) can be fitted as 
many bite-sized model fits, where the model fitted at each node consists of just tens of parameters. This makes 
graphical models scalable to largest datasets and most complex models!
"""
