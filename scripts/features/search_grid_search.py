"""
Feature: Search Grid Search
===========================

A classic method to perform model-fitting is a grid search, where the parameters of a model are divided on to a grid of
values and the likelihood of each set of parameters on this grid is sampled. For low dimensionality problems this
simple approach can be sufficient to locate high likelihood solutions, however it scales poorly to higher dimensional
problems.

**PyAutoFit** can perform a search grid search, which allows one to perform a grid-search over a subset of parameters
within a model, but use a non-linear search to fit for the other parameters. The parameters over which the grid-search
is performed are also included in the model fit and their values are simply confined to the boundaries of their grid
cell by setting these as the lower and upper limits of a `UniformPrior`.

The benefits of using a search grid search are:

 - For problems with complex and multi-model parameters spaces it can be difficult to robustly and efficiently perform
 model-fitting. If specific parameters are known to drive the multi-modality then sampling over a grid can ensure the
 parameter space of each individual model-fit is not multi-modal and therefore sampled more accurately and efficiently.

 - It can provide a goodness-of-fit measure (e.g. the Bayesian evidence) of many model-fits over the grid. This can
 provide additional insight into where the model does and does not fit the data well, in a way that a standard
 non-linear search does not.

 - The search grid search is embarrassingly parallel, and if sufficient computing facilities are available one can
 perform model-fitting faster in real-time than a single non-linear search. The **PyAutoFit** search grid search
 includes an option for parallel model-fitting via the Python `multiprocessing` module.

In this example we will demonstrate the search grid search feature, again using the example of fitting 1D Gaussian's
in noisy data.

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

Load data of a 1D Gaussian from a .json file in the directory 
`autofit_workspace/dataset/gaussian_x1_with_feature`.

This 1D data includes a small feature to the right of the central `Gaussian`. This feature is a second `Gaussian` 
centred on pixel 70. 
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_with_feature")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the data. 

The feature on pixel 70 is clearly visible.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.title("1D Gaussian Data With Feature at pixel 70.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Model__

Next, we create the model, which in this case corresponds to two `Gaussian`'s, one for the main signal seen in the
data and one for the feature on pixel 70.
"""
model = af.Collection(gaussian_main=af.ex.Gaussian, gaussian_feature=af.ex.Gaussian)

"""
The `info` attribute shows the model in a readable format, showing it contains two `Gaussian`'s.
"""
print(model.info)

"""
__Analysis__

Create the analysis which fits the model to the data.

It fits the data as the sum of the two `Gaussian`'s in the model.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

Fit the data using a single non-linear search, to demonstrate the behaviour of the fit before we invoke
the search grid search.
"""
search = af.DynestyStatic(
    path_prefix=path.join("features", "search_grid_search"),
    name="single_fit",
    nlive=100,
    maxcall=30000,
)

"""
To perform the fit with Dynesty, we pass it our model and analysis and we`re good to go!

Checkout the folder `autofit_workspace/output/features.search_grid_search/single_fit`, where the `NonLinearSearch` 
results, visualization and information can be found.

For test runs on my laptop it is 'hit or miss' whether the feature is fitted correctly. This is because although models
including the feature corresponds to the highest likelihood solutions, they occupy a small volume in parameter space
which the non linear search may miss. Furthemore, it is common for the model-fit to get stuck in local maxima where
both `Gaussian`'s go to a centre value of 50.0.

The fit can also take a very long time to run, therefore I limited `Dynesty` to 30000 iterations above.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

If you ran the fit above, you can now plot the result. 
"""
instance = result.max_log_likelihood_instance

gaussian_main = instance.gaussian_main.model_data_from(xvalues=np.arange(data.shape[0]))
gaussian_feature = instance.gaussian_feature.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_data = gaussian_main + gaussian_feature

plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_main, "--")
plt.plot(range(data.shape[0]), gaussian_feature, "--")
plt.title("Dynesty model fit to 1D Gaussian with feature dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search Grid Search__

We will now perform the search grid search. 

We will use the same `Dynesty` settings, but change its `name`.
"""
search = af.DynestyStatic(
    name="grid_fit",
    path_prefix=path.join("features", "search_grid_search"),
    nlive=100,
    maxcall=30000,
    number_of_cores=2,
    #   force_x1_cpu=True,  # ensures parallelizing over grid search works.
)

"""
To set up the search grid search we specify two additional settings:

`number_of_steps`: The number of steps in the grid search that are performedm which is set to 5 below. 
 
Because the prior on the parameter `centre` is a `UniformPrior` from 0.0 -> 100.0, this means the first grid search
will set the prior on the centre to be a `UniformPrior` from 0.0 -> 20.0. The second will run from 20.0 -> 40.0,
the third 40.0 -> 60.0, and so on.
   
`parallel`: If `True`, each grid search is performed in parallel on your laptop. 

`number_of_cores`: The number of cores the grid search will parallelize the run over. If `number_of_cores=1`, the
search is run in serial. For > 1 core, 1 core is reserved as a farmer, e.g., if `number_of_cores=4` then up to 3 
searches will be run in parallel. In case your laptop has limited hardware resources we do not run in parallel in 
this example by default, but feel free to change the option to `True` if you have a lot of CPUs and memory!
"""
grid_search = af.SearchGridSearch(search=search, number_of_steps=5, number_of_cores=1)

"""
We can now run the grid search.

This is where we specify the parameter over which the grid search is performed, in this case the `centre` of the 
`gaussian_feature` in our model.

On my laptop, each model fit performed by the grid search takes ~15000 iterations, whereas the fit above
required ~ 40000 iterations. Thus, in this simple example, the grid search did not speed up the overall analysis 
(unless it is run in parallel). However, more complex and realistic model-fitting problems, the grid search has the
potential to give huge performance improvements if used effectively.
"""
grid_search_result = grid_search.fit(
    model=model, analysis=analysis, grid_priors=[model.gaussian_feature.centre]
)

"""
This returns a `GridSearchResult`, which includes information on every model-fit performed on the grid.

Below, we print:
 
 - The central value of the `UniformPrior` on the `centre` of the gaussian_feature` for each fit performed on the
 grid search. 
 
 - The maximum log likelihood value of each of the 5 fits. 
 
 - The Bayesian evidence of each (this is accessible because we used a nested sampling algorithm).

You should see that the highest likelihood and evidence values correspond to run 4, where the `UniformPrior` on the
centre parameter ran from 60 -> 80 and therefore captured the true value of 70.0.
"""
print(grid_search_result.physical_centres_lists)
print(grid_search_result.log_likelihoods().native)
print(grid_search_result.log_evidences().native)

"""
We can also access the `best_samples` and their maximum likelihood instance.
"""
print(grid_search_result.best_samples)

instance = grid_search_result.best_samples.instance

print(instance.gaussian_main.centre)
print(instance.gaussian_main.normalization)
print(instance.gaussian_main.sigma)
print(instance.gaussian_feature.centre)
print(instance.gaussian_feature.normalization)
print(instance.gaussian_feature.sigma)

"""
By plotting the `best` instance we can confirm the grid search fitted the feature at pixel 70.
"""
gaussian_main = instance.gaussian_main.model_data_from(xvalues=np.arange(data.shape[0]))
gaussian_feature = instance.gaussian_feature.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_data = gaussian_main + gaussian_feature

plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_main, "--")
plt.plot(range(data.shape[0]), gaussian_feature, "--")
plt.title("Dynesty model fit to 1D Gaussian with feature dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
A multi-dimensional grid search can be easily performed by adding more parameters to the `grid_priors` input.

The fit below belows performs a 5x5 grid search over the `centres` of both `Gaussians`. This would take quite a long
time to run, so I've commented it out, but feel free to run it!
"""
# grid_search_result = grid_search.fit(
#     model=model,
#     analysis=analysis,
#     grid_priors=[model.gaussian_feature.centre, model.gaussian_main.centre],
# )

"""
Finish.
"""
