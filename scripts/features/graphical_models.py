"""
Feature: Graphical Models
=========================

The examples so far have focused on fitting one model to one dataset, for example fitting 1D profiles composed of
Gaussians to noisy 1D data. When multiple datasets were available each is fitted individually and their results
interpreted one-by-one.

However, for many problems we may have a large dataset and not be interested in how well the model fits each dataset
individually. Instead, we may wish to fit this model (or many similar models) to the full dataset and determine
the 'global' trends of the model across the datasets.

This can be done using graphical models, which compose and fit a model that has 'local' parameters specific to each
individual dataset but also higher-level model components that fit 'global' parameters of the model across the whole
dataset. This framework can be easily extended to fit datasets with different properties, complex models with different
topologies and has the functionality to allow it to be generalized to models with thousands of parameters.

In this example, we demonstrate the basic API for performing graphical modeling in **PyAutoFit** using the example of
simultaneously fitting 3 noisy 1D Gaussians. However, graphical models are an extensive feature and at the end of
this example we will discuss other options available in **PyAutoFit** for composing a fitting a graphical model.

The **HowToFit** tutorials contain a chapter dedicated to composing and fitting graphical models.

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

from os import path
import matplotlib.pyplot as plt

import autofit as af

"""
__Dataset__

We are going to build a graphical model that fits three datasets. 

We begin by loading noisy 1D data containing 3 Gaussian's.
"""
total_datasets = 3

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

    data_list.append(data)
    noise_map_list.append(noise_map)

"""
Now lets plot the data, including their error bars. One should note that we are fitting much lower signal-to-noise
datasets than usual.

Note that all three of these `Gaussian`'s were simulated using the same `centre` value. To demonstrate graphical 
modeling we will therefore fit a model where the `centre` a shared parameter across the fit to the 3 `Gaussian`s, 
therefore making it a global parameter.
"""
for dataset_index in range(total_datasets):
    xvalues = range(data_list[dataset_index].shape[0])

    plt.errorbar(
        x=xvalues,
        y=data_list[dataset_index],
        yerr=noise_map_list[dataset_index],
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Data #1.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()


"""
__Analysis__

They are much lower signal-to-noise than the Gaussian's in other examples. 

Graphical models extract a lot more information from lower quantity datasets, something we demonstrate explicitly 
in the **HowToFit** lectures on graphical models.

For each dataset we now create a corresponding `Analysis` class. By associating each dataset with an `Analysis`
class we are therefore associating it with a unique `log_likelihood_function`. 

If our dataset had many different formats which each required their own unique `log_likelihood_function`, it would 
be straight forward to write customized `Analysis` classes for each dataset.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__

We now compose the graphical model we will fit using the `Model` objects described in  the `overview` examples 
and chapter 1 of **HowToFit**.

We begin by setting up a shared prior for `centre`, which is set up this up as a single `GaussianPrior` that is 
passed to a unique `Model` for each `Gaussian`. This means all three `Gaussian`'s will be fitted wih the same 
value of `centre`.
"""
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

"""
We now set up three `Model`'s, each of which contain a `Gaussian` that is used to fit each of the 
datasets we loaded above.

All three of these `Model`'s use the `centre_shared_prior`. This means all three model-components use 
the same value of `centre` for every model composed and fitted by the `NonLinearSearch`, reducing the dimensionality 
of parameter space from N=9 (e.g. 3 parameters per Gaussian) to N=7.
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

Above, we composed a model consisting of three `Gaussian`'s with a shared `centre` prior. We also loaded three 
datasets which we intend to fit with each of these `Gaussians`, setting up each in an `Analysis` class that defines 
how the model is used to fit the data.

We now simply need to pair each model-component to each `Analysis` class, so that:

- `prior_model_0` fits `data_0` via `analysis_0`.
- `prior_model_1` fits `data_1` via `analysis_1`.
- `prior_model_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called a `AnalysisFactor`. 

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

We combine our `AnalysisFactors` into one, to compose the factor graph.

So, what is a factor graph?

A factor graph defines the graphical model we have composed. For example, it defines the different model components 
that make up our model (e.g. the three `Gaussian` classes) and how their parameters are linked or shared (e.g. that
each `Gaussian` has its own unique `normalization` and `sigma`, but a shared `centre` parameter.

This is what our factor graph looks like: 

The factor graph above is made up of two components:

- Nodes: these are points on the graph where we have a unique set of data and a model that is made up of a subset of 
our overall graphical model. This is effectively the `AnalysisFactor` objects we created above. 

- Links: these define the model components and parameters that are shared across different nodes and thus retain the 
same values when fitting different datasets.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
search = af.DynestyStatic(
    path_prefix="features", name="graphical_model", sample="rwalk"
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
This will fit the N=7 dimension parameter space where every Gaussian has a shared centre!

This is all expanded upon in the **HowToFit** chapter on graphical models, where we will give a more detailed
description of why this approach to model-fitting extracts a lot more information than fitting each dataset
one-by-one.

__Hierarchical Models__

A specific type of graphical model is a hierarchical model, where the shared parameter(s) of a graph are assumed
to be drawn from a common parent distribution. 

Fitting the datasets simultaneously enables better estimate of this global hierarchical distribution.

__Expectation Propagation__

For large datasets, a graphical model may have hundreds, thousands, or *hundreds of thousands* of parameters. The
high dimensionality of such a parameter space can make it inefficient or impossible to fit the model.

Fitting high dimensionality graphical models in **PyAutoFit** can use an Expectation Propagation (EP) framework to 
make scaling up feasible. This framework fits every dataset individually and pass messages throughout the graph to 
inform every fit the expected 
values of each parameter.

The following paper describes the EP framework in formal Bayesian notation:

https://arxiv.org/pdf/1412.4869.pdf

Hierarchical models can also be scaled up to large datasets via Expectation Propagation.
"""
