"""
Tutorial Optional: Hierarchical Expectation Propagation (EP)
============================================================

This optional tutorial gives an example of fitting a hierarchical model using EP.

The API is a straightforward combination of tutorials 3 and 4.
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

In this example, the three Gaussians have different centres, which are drawn from a parent Gaussian distribution
whose mean and scatter we aim to estimate.
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
By plotting the Gaussians we can just about make out that their centres are not all at pix 50, and are spreasd out
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

This uses a nearly identical for loop to the previous tutorial, however a shared `centre` is no longer used and each 
`Gaussian` is given its own prior for the `centre`. We will see next how this `centre` is used to construct the 
hierachical model.
"""

model_list = []

for model_index in range(len(data_list)):
    gaussian = af.Model(af.ex.Gaussian)

    # gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    #  gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    # gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

    gaussian.centre = af.TruncatedGaussianPrior(
        mean=50.0, sigma=20.0, lower_limit=0.0, upper_limit=100.0
    )
    gaussian.normalization = af.TruncatedGaussianPrior(
        mean=3.0, sigma=5.0, lower_limit=0.0
    )
    gaussian.sigma = af.TruncatedGaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)

    model_list.append(gaussian)

"""
__Analysis Factors__

Now we have our `Analysis` classes and model components, we can compose our `AnalysisFactor`'s.

The hierarchical model fit uses EP, therefore we again supply each `AnalysisFactor` its own `search` and `name`.
"""
dynesty = af.DynestyStatic(nlive=100, sample="rwalk")

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):
    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=dynesty, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)


"""
__Model__

We now compose the hierarchical model that we fit, using the individual Gaussian model components we created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `centre` of each individual `Gaussian` dataset is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""

hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.TruncatedGaussianPrior(
        mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0
    ),
    sigma=af.TruncatedGaussianPrior(
        mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
    ),
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

Note that in previous tutorials, when we created the `FactorGraphModel` we only passed the list of `AnalysisFactor`'s,
which contained the necessary information on the model create the factor graph that was fitted. The `AnalysisFactor`'s
were created before we composed the `HierachicalFactor`, which is why we need to pass it separate when composing the
factor graph.
"""

factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
__Model Fit__


"""
laplace = af.LaplaceOptimiser()

ep_result = factor_graph.optimise(
    laplace,
    paths=af.DirectoryPaths(
        name=path.join(
            "howtofit", "chapter_graphical_models", "tutorial_4_hierarchical"
        )
    ),
    ep_history=af.EPHistory(kl_tol=1.0),
    max_steps=5,
)
