"""
Tutorial 1: Individual Models
=============================

In many examples, we fit one model to one dataset. For many problems, we may have a large dataset and are not
interested in how well the model fits each individual dataset. Instead, we want to know how the model fits the full
dataset, so that we can determine "global" trends of how the model fits the data.

These tutorials show you how to compose and fit hierarchical models to large datasets, which fit many individual
models to each dataset. However, all parameters in the model are linked together, enabling global inference of the
model over the full dataset. This can extract a significant amount of extra information from large datasets, which
fitting each dataset individually does not.

Fitting a hierarchical model uses a "graphical model", which is a model that is simultaneously fitted to every
dataset simultaneously. The graph expresses how the parameters of every individual model is paired with each dataset
and how they are linked to every other model parameter. Complex graphical models fitting a diversity of different
datasets and non-trivial model parameter linking is possible and common.

This chapter will start by fitting a simple graphical model to a dataset of noisy 1D Gaussians. The Gaussians all
share the same `centre`, meaning that a graphical model can be composed where there is only a single global `centre`
shared by all Gaussians.

However, before fitting a graphical model, we will first fit each Gaussian individually and combine the inference
on the `centre` after every fit is complete. This will give us an estimate of the `centre` that we can compare to
the result of the graphical model in tutorial 2.

__Real World Example__

Hierarchical models are often used to determine effective drug treatments across a sample of patients distributed over
many hospitals. Trying to do this on each individual hospital dataset is not ideal, as the number of patients in each
hospital is small and the treatment may be more or less effective in some hospitals than others. Hierarchical models
can extract the global trends of how effective the treatment is across the full population of patients.

In healthcare, there may also be many datasets available, with different formats that require slightly different models
to fit them. The high levels of customization possible in model composition and defining the analysis class mean
that fitting diverse datasets with hierarchical models is feasible. This also means that a common problem in healthcare
data, missing data, can be treated in a statistically robust manner.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np

import autofit as af
import autofit.plot as aplt

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and 
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you have 
seen and used elsewhere throughout the workspace.

__Model__

Our model is a single `Gaussian`. 

We put this in a `Collection` so that when we extend the model in later tutorials we use the same API throughout
all tutorials.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

"""
__Data__

We quickly set up the name of each dataset, which is used below for loading the datasets.

The dataset contains 10 Gaussians, but for speed we'll fit just 5. You can change this to 10 to see how the result
changes with more datasets.
"""
total_datasets = 5

dataset_name_list = []

for dataset_index in range(total_datasets):
    dataset_name_list.append(f"dataset_{dataset_index}")

"""
For each 1D Gaussian dataset we now set up the correct path, load it, and plot it. 

Notice how much lower the signal-to-noise is than you are used too, you probably find it difficult to estimate 
the centre of some of the Gaussians by eye!
"""
for dataset_name in dataset_name_list:
    """
    Load the dataset from the `autofit_workspace/dataset` folder.
    """

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    af.ex.plot_profile_1d(
        xvalues=np.arange(data.shape[0]),
        profile_1d=data,
        title=dataset_name,
        ylabel="Data Values",
        color="k",
    )

"""
__Model Fits (one-by-one)__

For every dataset we now create an `Analysis` and fit it with a `Gaussian`.

The `Result` is stored in the list `result_list`.
"""
result_list = []

for dataset_name in dataset_name_list:
    """
    Load the dataset from the `autofit_workspace/dataset` folder.
    """
    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    """
    For each dataset create a corresponding `Analysis` class.
    """
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    """
    Create the `DynestyStatic` non-linear search and use it to fit the data.

    We use custom dynesty settings which ensure the posterior is explored fully and that our error estimates are robust.
    """
    dynesty = af.DynestyStatic(
        name="global_model",
        path_prefix=path.join(
            "howtofit", "chapter_graphical_models", "tutorial_1_individual_models"
        ),
        unique_tag=dataset_name,
        nlive=200,
        dlogz=1e-4,
        sample="rwalk",
        walks=10,
    )

    print(
        f"Dynesty has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/chapter_graphica_models/tutorial_1_individual_models/{dataset_name} for live \n"
        f"output of the results. This Jupyter notebook cell with progress once Dynesty has completed, this could take a \n"
        f"few minutes!"
    )

    result_list.append(dynesty.fit(model=model, analysis=analysis))

"""
__Results__

Checkout the output folder, you should see five new sets of results corresponding to our Gaussian datasets.

In the `model.results` file of each fit, it will be clear that the `centre` value of every fit (and the other 
parameters) have much larger errors than other **PyAutoFit** examples due to the low signal to noise of the data.

The `result_list` allows us to plot the median PDF value and 3.0 confidence intervals of the `centre` estimate from
the model-fit to each dataset.
"""
import matplotlib.pyplot as plt

samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
ue3_instances = [samp.errors_at_upper_sigma(sigma=3.0) for samp in samples_list]
le3_instances = [samp.errors_at_lower_sigma(sigma=3.0) for samp in samples_list]

mp_centres = [instance.gaussian.centre for instance in mp_instances]
ue3_centres = [instance.gaussian.centre for instance in ue3_instances]
le3_centres = [instance.gaussian.centre for instance in le3_instances]

plt.errorbar(
    x=[f"Gaussian {index}" for index in range(total_datasets)],
    y=mp_centres,
    marker=".",
    linestyle="",
    yerr=[le3_centres, ue3_centres],
)
plt.xticks(rotation=90)
plt.show()
plt.close()

"""
These model-fits are consistent with a range of `centre` values. 

We can show this by plotting the 1D and 2D PDF's of each model fit
"""

for samples in samples_list:
    search_plotter = aplt.DynestyPlotter(samples=samples)
    search_plotter.cornerplot()

"""
We can also print the values of each centre estimate, including their estimates at 3.0 sigma.

Note that above we used the samples to estimate the size of the errors on the parameters. Below, we use the samples to 
get the value of the parameter at these sigma confidence intervals.
"""
u1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
l1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

u1_centres = [instance.gaussian.centre for instance in u1_instances]
l1_centres = [instance.gaussian.centre for instance in l1_instances]

u3_instances = [samp.values_at_upper_sigma(sigma=3.0) for samp in samples_list]
l3_instances = [samp.values_at_lower_sigma(sigma=3.0) for samp in samples_list]

u3_centres = [instance.gaussian.centre for instance in u3_instances]
l3_centres = [instance.gaussian.centre for instance in l3_instances]

for index in range(total_datasets):
    print(f"Centre estimate of Gaussian dataset {index}:\n")
    print(
        f"{mp_centres[index]} ({l1_centres[index]} {u1_centres[index]}) [1.0 sigma confidence interval]"
    )
    print(
        f"{mp_centres[index]} ({l3_centres[index]} {u3_centres[index]}) [3.0 sigma confidence interval] \n"
    )

"""
__Estimating the Centre__

So how might we estimate our global `centre` value? 

A simple approach takes the weighted average of the value inferred by all five fits above.
"""
ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_centres = [instance.gaussian.centre for instance in ue1_instances]
le1_centres = [instance.gaussian.centre for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres, le1_centres)]

values = np.asarray(mp_centres)
sigmas = np.asarray(error_list)

weights = 1 / sigmas ** 2.0
weight_averaged = np.sum(1.0 / sigmas ** 2)

weighted_centre = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error = 1.0 / np.sqrt(weight_averaged)

print(
    f"Weighted Average Centre Estimate = {weighted_centre} ({weighted_error}) [1.0 sigma confidence intervals]"
)

"""
__Posterior Multiplication__

An alternative and more accurate way to combine each individual inferred centre is multiply their posteriors together.

In order to do this, a smooth 1D profile must be fit to the posteriors via a Kernel Density Estimator (KDE).

[There is currently no support for posterior multiplication and an example illustrating this is currently missing 
from this tutorial. However, I will discuss KDE multiplication throughout these tutorials to give the reader context 
for how this approach to parameter estimation compares to graphical models.]

__Wrap Up__

Lets wrap up the tutorial. The methods used above combine the results of different fits and estimate a global 
value of `centre` alongside estimates of its error. 

In this tutorial, we fitted just 5 datasets. Of course, we could easily fit more datasets, and we would find that
as we added more datasets our estimate of the global centre would become more precise.

In the next tutorial, we will compare this result to one inferred via a graphical model. 

Tutorial 2: Graphical Models
============================

We have fitted a dataset containing 5 noisy 1D Gaussian which had a shared `centre` value. We estimated
the `centre` by fitting each dataset individually and combining the value of the `centre` inferred by each fit into
an overall estimate, using a weighted average.

Graphical models use a different approach. They are a single model that is fitted to the entire dataset simultaneously. 
The model includes specific model component for every individual 1D Gaussian in the sample. However, the graphical 
model also has shared parameters between these individual model components.

This example fits a graphical model using the same sample fitted in the previous tutorial, consisting of many 1D 
Gaussians. However, whereas previously the `centre` of each Gaussian was a free parameter in each fit, in the graphical 
model there is only a single parameter for the `centre` shared by all 1D Gaussians.

This graphical model creates a non-linear parameter space with parameters for every Gaussian in our sample. For 5
Gaussians each with their own model parameters but a single shared centre:

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

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you 
have seen and used elsewhere throughout the workspace.

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

All of these models use the `centre_shared_prior`, meaning that all model-components use the same value of `centre` 
for every individual model component. 

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

We now simply pair each model-component to each `Analysis` class, so that:

- `gaussian_0` fits `data_0` via `analysis_0`.
- `gaussian_1` fits `data_1` via `analysis_1`.
- `gaussian_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called an `AnalysisFactor`. 

This term denotes that we are composing a graphical model, which is commonly called a 'factor graph'. A  factor 
defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between these 
different nodes then define the global model we are fitting.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We now combine our `AnalysisFactor`'s to compose a factor graph.

What is a factor graph? A factor graph defines the graphical model's graph. For example, it defines the different 
model components that make up our model (e.g. the individual `Gaussian` classes) and how their parameters are linked or 
shared (e.g. that each `Gaussian` has its own unique `normalization` and `sigma`, but a shared `centre` parameter).

This is what our factor graph looks like (visualization of graphs not implemented yet): 

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
    nlive=200,
    dlogz=1e-4,
    sample="rwalk",
    walks=10,
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

Tutorial 3: Graphical Benefits
==============================

In the previous tutorials, we fitted a dataset containing 5 noisy 1D Gaussian which had a shared `centre` value and
compared different approaches to estimate the shared `centre`. This included a simple approach fitting each dataset
one-by-one and estimating the centre via a weighted average or posterior multiplication and a more complicated
approach using a graphical model.

The estimates were consistent with one another, making it hard to justify the use of the more complicated graphical
model. However, the model fitted in the previous tutorial was extremely simple, and by making it slightly more complex
we will show the benefits of the graphical model.

__The Model__

In this tutorial, each dataset now contains two Gaussians, and they all have the same shared centres, located at
pixels 40 and 60.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autofit.plot as aplt

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

Note that we are loading a new dataset called `gaussian_x2__offset_centres`.
"""
total_datasets = 5

dataset_name_list = []
data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x2__offset_centres", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
By plotting the datasets we see that each dataset contains two Gaussians. 

Their centres are offset from one another and not located at pixel 50, like in the previous tutorials. 

As discussed above, the Gaussians in every dataset are in facted centred at pixels 40 and 60.
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
__Model (one-by-one)__

We are first going to fit each dataset one by one.

Our model therefore now has two 1D `Gaussian`'s.

To remove solutions where the Gaussians flip locations and fit the other Gaussian, we set uniform priors on the
`centre`'s which ensures one Gaussian stays on the left side of the data (fitting the Gaussian at pixel 40) 
whilst the other stays on the right (fitting the Gaussian at pixel 60).
"""
gaussian_0 = af.Model(af.ex.Gaussian)

gaussian_0.centre = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

gaussian_1 = af.Model(af.ex.Gaussian)

gaussian_1.centre = af.UniformPrior(lower_limit=50.0, upper_limit=100.0)

model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

"""
__Model Fits (one-by-one)__

For every dataset we now create an `Analysis` and fit it with a `Gaussian`.

The `Result` is stored in the list `result_list`.
"""
result_list = []

for i, analysis in enumerate(analysis_list):
    """
    Create the `DynestyStatic` non-linear search and use it to fit the data.

    We use custom dynesty settings which ensure the posterior is explored fully and that our error estimates are robust.
    """
    search = af.DynestyStatic(
        name=f"individual_fit_{i}",
        path_prefix=path.join(
            "howtofit", "chapter_graphical_models", "tutorial_3_graphical_benefits"
        ),
        nlive=200,
        dlogz=1e-4,
        sample="rwalk",
        walks=10,
    )

    print(
        f"Dynesty has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/chapter_graphical_models/tutorial_3_graphical_benefits/{dataset_name} for live \n"
        f"output of the results. This Jupyter notebook cell with progress once Dynesty has completed, this could take a \n"
        f"few minutes!"
    )

    result_list.append(search.fit(model=model, analysis=analysis))

"""
__Centre Estimates (Weighted Average)__

We can now compute the centre estimate of both Gaussians, including their errors, from the individual model fits
performed above.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]

mp_centres_0 = [instance.gaussian_0.centre for instance in mp_instances]
mp_centres_1 = [instance.gaussian_1.centre for instance in mp_instances]

ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_centres_0 = [instance.gaussian_0.centre for instance in ue1_instances]
le1_centres_0 = [instance.gaussian_0.centre for instance in le1_instances]

error_0_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres_0, le1_centres_0)]

values_0 = np.asarray(mp_centres_0)
sigmas_0 = np.asarray(error_0_list)

weights_0 = 1 / sigmas_0 ** 2.0
weight_averaged_0 = np.sum(1.0 / sigmas_0 ** 2)

weighted_centre_0 = np.sum(values_0 * weights_0) / np.sum(weights_0, axis=0)
weighted_error_0 = 1.0 / np.sqrt(weight_averaged_0)

ue1_centres_1 = [instance.gaussian_1.centre for instance in ue1_instances]
le1_centres_1 = [instance.gaussian_1.centre for instance in le1_instances]

error_1_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres_1, le1_centres_1)]

values_1 = np.asarray(mp_centres_1)
sigmas_1 = np.asarray(error_1_list)

weights_1 = 1 / sigmas_1 ** 2.0
weight_averaged_1 = np.sum(1.0 / sigmas_1 ** 2)

weighted_centre_1 = np.sum(values_1 * weights_1) / np.sum(weights_1, axis=0)
weighted_error_1 = 1.0 / np.sqrt(weight_averaged_1)

print(
    f"Centre 0 via Weighted Average: {weighted_centre_0} ({weighted_error_0}) [1.0 sigma confidence intervals]"
)
print(
    f"Centre 1 via Weighted Average: {weighted_centre_1} ({weighted_error_1}) [1.0 sigma confidence intervals]"
)

"""
The estimate of the centres is not accurate, with both estimates well offset from the input values of 40 and 60.

We will next show that the graphical model offers a notable improvement, but first lets consider why this
approach is suboptimal.

The most important difference between this model and the model fitted in the previous tutorial is that there are now
two shared parameters we are trying to estimate, which are degenerate with one another.

We can see this by inspecting the probability distribution function (PDF) of the fit, placing particular focus on the 
2D degeneracy between the Gaussians centres. 
"""
search_plotter = aplt.DynestyPlotter(samples=result_list[0].samples)
search_plotter.cornerplot()

"""
The problem is that the simple approach of taking a weighted average does not capture the curved banana-like shape
of the PDF between the two centres. This leads to significant error over estimation and biased inferences on the centre.

__Discussion__

Let us now consider other downsides of fitting each dataset one-by-one, from a statistical perspective. We 
will contrast these to the graphical model later in the tutorial:

1) By fitting each dataset one-by-one this means that each model-fit fails to fully exploit the information we know 
about the global model. We know that there are only two single shared values of `centre` across the full dataset 
that we want to estimate. However, each individual fit has its own `centre` value which is able to assume different 
values than the `centre` values used to fit the other datasets. This means that large degeneracies between the two 
centres are present in each model-fit.

By not fitting our model as a global model, we do not maximize the amount of information that we can extract from the 
dataset as a whole. If a model fits dataset 1 poorly, this should be reflected in how we interpret how well the model 
fits datasets 2 and 3. Our non-linear search should have a global view of how well the model fits the whole dataset. 
This is the crucial aspect of fitting each dataset individually that we miss, and what a graphical model addresses.

2) When we combined the result to estimate the global `centre` value via a weighted average, we marginalized over 
the samples in 1D. As showed above, when there are strong degeneracies between models parameters the information on 
the covariance between these parameters is lost when computing the global `centre`. This increases the inferred 
uncertainties. A graphical model performs no such 1D marginalization and therefore fully samples the
parameter covariances.

3) In Bayesian inference it is important that we define priors on all of the model parameters. By estimating the 
global `centre` after the model-fits are completed it is unclear what prior the global `centre` actually has! We
actually defined the prior five times -- once for each fit -- which is not a well defined prior. In a graphical model 
the prior is clearly defined.

What would have happened if we had estimate the shared centres via 2D posterior multiplication using a KDE? We
will discuss this at the end of the tutorial after fitting a graphical model.

__Model (Graphical)__

We now compose a graphical model and fit it.

Our model now consists of two Gaussians with two `centre_shared_prior` variables, such that the same centres are
used for each Gaussians across all datasets. 

We again restrict one Gaussian's centre between pixels 0 -> 50 and the other 50 -> 100 to remove solutions where
the Gaussians flip location.
"""
centre_0_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
centre_1_shared_prior = af.UniformPrior(lower_limit=50.0, upper_limit=100.0)

"""
We now set up a list of `Model`'s, each of which contain two `Gaussian`'s that are used to fit each of the datasets 
loaded above.

All of these `Model`'s use the `centre_shared_prior`'s abpve. This means all model-components use the same value 
of `centre` for every model composed and fitted. 

For a fit to five datasets (each using two Gaussians), this reduces the dimensionality of parameter space 
from N=30 (e.g. 6 parameters per pair of Gaussians) to N=22 (e.g. 10 `sigma`'s 10 `normalizations` and 2 `centre`'s).
"""
model_list = []

for model_index in range(len(data_list)):
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)

    gaussian_0.centre = centre_0_shared_prior  # This prior is used by all Gaussians!
    gaussian_1.centre = centre_1_shared_prior  # This prior is used by all Gaussians!

    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    model_list.append(model)

"""
__Analysis Factors__

We again create the graphical model using `AnalysisFactor` objects.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

The analysis factors are then used to create the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
The factor graph model can again be printed via the `info` attribute, which shows that there are two shared
parameters across the datasets.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and use it to the fit the factor graph, again using its `global_prior_model` 
property.
"""
search = af.DynestyStatic(
    path_prefix=path.join("howtofit", "chapter_graphical_models"),
    name="tutorial_3_graphical_benefits",
    sample="rwalk",
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same structure of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
We can now inspect the inferred `centre` values and compare this to the values estimated above via a weighted average.  

(The errors of the weighted average is what was estimated for a run on my PC, yours may be slightly different!)
"""
centre_0 = result.samples.median_pdf()[0].gaussian_0.centre

u1_error_0 = result.samples.values_at_upper_sigma(sigma=1.0)[0].gaussian_0.centre
l1_error_0 = result.samples.values_at_lower_sigma(sigma=1.0)[0].gaussian_0.centre

u3_error_0 = result.samples.values_at_upper_sigma(sigma=3.0)[0].gaussian_0.centre
l3_error_0 = result.samples.values_at_lower_sigma(sigma=3.0)[0].gaussian_0.centre

centre_1 = result.samples.median_pdf()[0].gaussian_1.centre

u1_error_1 = result.samples.values_at_upper_sigma(sigma=1.0)[0].gaussian_1.centre
l1_error_1 = result.samples.values_at_lower_sigma(sigma=1.0)[0].gaussian_1.centre

u3_error_1 = result.samples.values_at_upper_sigma(sigma=3.0)[0].gaussian_1.centre
l3_error_1 = result.samples.values_at_lower_sigma(sigma=3.0)[0].gaussian_1.centre

print(
    f"Centre 0 via Weighted Average: 29.415828686393333 (15.265325182888517) [1.0 sigma confidence intervals] \n"
)
print(
    f"Centre 1 via Weighted Average: 54.13825075629124 (2.3460686758693234) [1.0 sigma confidence intervals] \n"
)

print(
    f"Inferred value of Gaussian 0's shared centre via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{centre_0} ({l1_error_0} {u1_error_0}) ({u1_error_0 - l1_error_0}) [1.0 sigma confidence intervals]"
)
print(
    f"{centre_0} ({l3_error_0} {u3_error_0}) ({u3_error_0 - l3_error_0}) [3.0 sigma confidence intervals]"
)

print(
    f"Inferred value of Gaussian 1's shared centre via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{centre_1} ({l1_error_1} {u1_error_1}) ({u1_error_1 - l1_error_1}) [1.0 sigma confidence intervals]"
)
print(
    f"{centre_1} ({l3_error_1} {u3_error_1}) ({u3_error_1 - l3_error_1}) [3.0 sigma confidence intervals]"
)

"""
As expected, using a graphical model allows us to infer a more precise and accurate model.

__Discussion__

Unlike a fit to each dataset one-by-one, the graphical model:

1) Infers a PDF on the global centre that fully accounts for the degeneracies between the models fitted to different 
datasets. This reduces significantly the large 2D degeneracies between the two centres we saw when inspecting the PDFs 
of each individual fit.

2) Fully exploits the information we know about the global model, for example that the centre of every Gaussian in every 
dataset is aligned. Now, the fit of the Gaussian in dataset 1 informs the fits in datasets 2 and 3, and visa versa.

3) Has a well defined prior on the global centre, instead of 5 independent priors on the centre of each dataset.

__Posterior Multiplication__

What if we had combined the results of the individual model fits using 2D posterior multiplication via a KDE?

This would produce an inaccurate estimate of the error, because each posterior contains the prior on the centre five 
times which given the properties of this model should not be repeated.

However, it is possible to convert each posterior to a likelihood (by dividing by its prior), combining these 5
likelihoods to form a joint likelihood via 2D KDE multiplication and then insert just one prior back (again using a 2D
KDE) at the end to get a posterior which does not have repeated priors. 

This posterior, in theory, should be equivalent to the graphical model, giving the same accurate estimates of the
centres with precise errors. The process extracts the same information, fully accounting for the 2D structure of the
PDF between the two centres for each fit.

However, in practise, this will likely not work well. Every time we use a KDE to represent and multiply a posterior, we 
make an approximation which will impact our inferred errors. The removal of the prior before combining the likelihood
and reinserting it after also introduces approximations, especially because the fit performed by the non-linear search
is informed by the prior. 

Crucially, whilst posterior multiplication can work in two dimensions, for models with many more dimensions and 
degeneracies between parameters that are in 3D, 4D of more dimensions it will introduce more and more numerical
inaccuracies.

A graphical model fully samples all of the information a large dataset contains about the model, without making 
such large approximation. Therefore, irrespective of how complex the model gets, it extracts significantly more 
information contained in the dataset.

__Wrap Up__

In this tutorial, we demonstrated the strengths of a graphical model over fitting each dataset one-by-one. 

We argued that irrespective of how one may try to combine the results of many individual fits, the approximations that 
are made will always lead to a suboptimal estimation of the model parameters and fail to fully extract all information
from the dataset. 

We argued that for high dimensional complex models a graphical model is the only way to fully extract all of the 
information contained in the dataset.

In the next tutorial, we will consider a natural extension of a graphical model called a hierarchical model.

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

This is called a hierarchical model, whihc we fit in this tutorial. The `centre` of each 1D Gaussian is now no
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

