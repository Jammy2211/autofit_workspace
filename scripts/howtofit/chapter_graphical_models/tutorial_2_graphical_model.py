"""
Tutorial 2: Graphical Models
============================

In the previous tutorial, we fitted a dataset containing 3 noisy 1D Gaussian which had a shared and global value of
`centre`. We attempted to estimate this global `centre` value by fitting each dataset individually with a 1D `Gaussian`. 
We then combined the inferred `centre` values of each fit to estimate the global `centre`, by either taking the mean 
values of each `centre` or combining the fits into a joint PDF.

We concluded that estimating the global `centre` in these ways was suboptimal, and that we were better of fitting for
the global `centre` in our model by fitting all 3 datasets simultaneously. In this tutorial we will do this by 
composing a graphical model.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from os import path

"""
We'll use the `Analysis` class of chapter 1, which includes `visualize` and `save_attributes_for_aggregator` methods.
"""
from analysis import Analysis

"""
For each dataset we now set up the correct path and load it. Whereas in the previous tutorial we fitted each dataset 
one-by-one, in this tutorial we will instead store each dataset in a list so that we can set up a single model-fit 
that fits the 3 datasets simultaneously.

NOTE: In this tutorial we explicitly write code for loading each dataset and store each dataset as their own Python
variable (e.g. data_0, data_1, data_2, etc.). We do not use a for loop or a list to do this (like we did in previous 
tutorials), even though this would be syntactically cleaner code. This is to make the API for setting up a graphical 
model in this tutorial clear and explicit; in the next tutorial we will introduce  the **PyAutoFit** API for setting 
up a graphical model for large datasets concisely.

[FOR RICH: For graphical models, the use of the `Analysis` class will cause memory issues for large datasets. This is
because every instance of the `Analysis` class is made before we run the graphical model. For large datasets this will
cripple memory. We should discussing how we can avoid this.]
"""
dataset_path = path.join("dataset", "example_1d")

dataset_0_path = path.join(dataset_path, "gaussian_x1_0__low_snr")
data_0 = af.util.numpy_array_from_json(file_path=path.join(dataset_0_path, "data.json"))
noise_map_0 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_0_path, "noise_map.json")
)

dataset_1_path = path.join(dataset_path, "gaussian_x1_1__low_snr")
data_1 = af.util.numpy_array_from_json(file_path=path.join(dataset_1_path, "data.json"))
noise_map_1 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_1_path, "noise_map.json")
)

dataset_2_path = path.join(dataset_path, "gaussian_x1_2__low_snr")
data_2 = af.util.numpy_array_from_json(file_path=path.join(dataset_2_path, "data.json"))
noise_map_2 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_2_path, "noise_map.json")
)

"""
For each dataset we now create a corresponding `Analysis` class. 

By associating each dataset with an `Analysis` class we are therefore also setting up the `log_likelihood_function` 
used to fit it. In the next tutorial we will introduce a graphical model that fits shared model components to datasets 
that are of different formats, such that each fit is performed using a different `log_likelihood_function` and 
therefore `Analysis` class.
"""
analysis_0 = Analysis(data=data_0, noise_map=noise_map_0)
analysis_1 = Analysis(data=data_1, noise_map=noise_map_1)
analysis_2 = Analysis(data=data_2, noise_map=noise_map_2)

"""
We now compose the graphical model we will fit. This uses the `PriorModel` and `CollectionPriorModel` objects you 
are now familiar with.
"""
from autofit import graphical as g
import profiles as p

"""
We begin by setting up a shared prior for `centre`. 

We set up this up as a single `GaussianPrior` which will be passed to separate `PriorModel`'s for each `Gaussian` used 
to fit each dataset. 
"""
centre_shared_prior = af.GaussianPrior(mean=10.0, sigma=10.0)

"""
We now set up three `CollectionPriorModel`'s, each of which contain a `Gaussian` that is used to fit each of the 
datasets we loaded above.

All three of these `CollectionPriorModel`'s use the `centre_shared_prior`. This means all three model-components use 
the same value of `centre` for every model composed and fitted by the `NonLinearSearch`, reducing the dimensionality 
of parameter space from N=9 (e.g. 3 parameters per Gaussian) to N=7.
"""
gaussian_0 = af.PriorModel(p.Gaussian)
gaussian_0.centre = af.GaussianPrior(mean=50, sigma=20)
gaussian_0.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
gaussian_0.sigma = centre_shared_prior  # This prior is used by all 3 Gaussians!

prior_model_0 = af.CollectionPriorModel(gaussian=gaussian_0)

gaussian_1 = af.PriorModel(p.Gaussian)
gaussian_1.centre = af.GaussianPrior(mean=50, sigma=20)
gaussian_1.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
gaussian_1.sigma = centre_shared_prior  # This prior is used by all 3 Gaussians!

prior_model_1 = af.CollectionPriorModel(gaussian=gaussian_1)

gaussian_2 = af.PriorModel(p.Gaussian)
gaussian_2.centre = af.GaussianPrior(mean=50, sigma=20)
gaussian_2.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
gaussian_2.sigma = centre_shared_prior  # This prior is used by all 3 Gaussians!

prior_model_2 = af.CollectionPriorModel(gaussian=gaussian_2)

"""
Above, we composed a model consisting of three `Gaussian`'s with a shared `centre` prior. We also loaded three datasets
which we intend to fit with each of these `Gaussians`, setting up each in an `Analysis` class that defines how the 
model is used to fit the data.

We now simply need to pair each model-component to each `Analysis` class, so that **PyAutoFit** knows that: 

- `prior_model_0` fits `data_0` via `analysis_0`.
- `prior_model_1` fits `data_1` via `analysis_1`.
- `prior_model_2` fits `data_2` via `analysis_2`.

The point where a `PriorModel` and `Analysis` class meet is called a `ModelFactor`. 

This term is used to denote that we are composing a graphical model, which is commonly termed a 'factor graph'. A 
factor defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between 
these different nodes then define the global model we are fitting.
"""
model_factor_0 = g.ModelFactor(prior_model=prior_model_0, analysis=analysis_0)
model_factor_1 = g.ModelFactor(prior_model=prior_model_1, analysis=analysis_1)
model_factor_2 = g.ModelFactor(prior_model=prior_model_2, analysis=analysis_2)

"""
We combine our `ModelFactors` into one, to compose the factor graph.
"""
factor_graph = g.FactorGraphModel(model_factor_0, model_factor_1, model_factor_2)

"""
So, what is a factor graph?

A factor graph defines the graphical model we have composed. For example, it defines the different model components 
that make up our model (e.g. the three `Gaussian` classes) and how their parameters are linked or shared (e.g. that
each `Gaussian` has its own unique `intensity` and `centre`, but a shared `sigma` parameter.

This is what our factor graph looks like: 

The factor graph above is made up of two components:

- Nodes: these are points on the graph where we have a unique set of data and a model that is made up of a subset of 
our overall graphical model. This is effectively the `ModelFactor` objects we created above. 

- Links: these define the model components and parameters that are shared across different nodes and thus retain the 
same values when fitting different datasets.
"""
from autofit.graphical import optimise

laplace = optimise.LaplaceFactorOptimiser()
collection = factor_graph.optimise(laplace)

print(collection)

"""
Finish.
"""
