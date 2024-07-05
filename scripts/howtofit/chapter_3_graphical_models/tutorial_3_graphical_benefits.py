"""
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
        f"The non-linear search has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/chapter_graphical_models/tutorial_3_graphical_benefits/{dataset_name} for live \n"
        f"output of the results. This Jupyter notebook cell with progress once search has completed, this could take a \n"
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

weights_0 = 1 / sigmas_0**2.0
weight_averaged_0 = np.sum(1.0 / sigmas_0**2)

weighted_centre_0 = np.sum(values_0 * weights_0) / np.sum(weights_0, axis=0)
weighted_error_0 = 1.0 / np.sqrt(weight_averaged_0)

ue1_centres_1 = [instance.gaussian_1.centre for instance in ue1_instances]
le1_centres_1 = [instance.gaussian_1.centre for instance in le1_instances]

error_1_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres_1, le1_centres_1)]

values_1 = np.asarray(mp_centres_1)
sigmas_1 = np.asarray(error_1_list)

weights_1 = 1 / sigmas_1**2.0
weight_averaged_1 = np.sum(1.0 / sigmas_1**2)

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
plotter = aplt.NestPlotter(samples=result_list[0].samples)
plotter.corner_cornerpy()

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
"""
