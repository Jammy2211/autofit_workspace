"""
Tutorial 1: Individual Modelss
=========================

In chapter 1, we focused on fitting one model to one dataset. We inspected the results of those individual model-fits
and demonstrated that when necessary we are able to fit more complex models, for example 1D profiles composed of 2 or 3
Gaussians / Exponentials.

For many problems we may have a large dataset and not be interested in how well the model fits each individual dataset.
Instead, our interest is fitting this dataset in its entirety, to determine 'global' trends of how the model fits the
data.

This chapter of **HowToFit** shows you how to compose and fit graphical models to these large datasets. We'll see that
a graphical model links together the many individual models that fit each dataset, but that it also contains shared
parameters that describe the global behaviour of how the model fits the full dataset.

Lets consider a real world example. In a healthcare setting, we may fit a model to some data and be able to show
that a certain treatment helps an individual patient recover from illness. However, for us to claim this treatment is
effective more generally, we would have to fit this model to a large patient dataset. One could imagine that we would
compose and fit many individual models to the dataset on each patient, but now link them together with shared parameters
describing how effective the treatment is across the full global population.

Medical datasets may contain data on thousands of patients, and we may therefore need to fit very complex models with
many thousands of parameters to determine global trends. We may have different datasets on different patients, meaning
that we need high levels of customization in the models we compose and fitting procedures that we apply. The
**PyAutoFit** graphical modeling framework provides the tools we need to do these tasks in a computationally tractable
manner.

This chapter illustrates the problem using the toy model of fitting noisy 1D datasets. The dataset contains up to 10
noisy 1D Gaussians, which all have the same value of `centre=50.0`. The centre is therefore the shared parameter we want
to infer across the full dataset -- it is the global aspect of our model we are actually interested in!

This tutorial does not use graphical models. Instead, it attempts to estimate the `centre` is the simpliest way
imaginable, by fitting each dataset one-by-one and combining the results after all model-fitting is complete. In
tutorial 2, we will do the fitting 'properly' using graphical models, and compare to our estimate in this tutorial to
see how much our estimate of the `centre` improves.
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

For every dataset we now create an `Analysis` and use `Dynesty` to fit it with a `Gaussian`.

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

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

weighted_centre = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error = 1.0 / np.sqrt(weight_averaged)

print(
    f"Weighted Average Centre Estimate = {weighted_centre} ({weighted_error}) [1.0 sigma confidence intervals]"
)

"""
__Posterior Multiplication__

An alternative and more accurate way to combine each individual infered centre is multiply their posteriors together.

In order to do this, a smooth 1D profile must be fit to the posteriors via a Kernel Density Estimator (KDE).

[**PyAutoFit** does not currently support posterior multiplication and an example illustrating this is currently
missing from this tutorial. However, I will discuss KDE multiplication throughout these tutorials to give the
reader context for how this approach to parameter estimation compares to graphical models.]

__Wrap Up__

Lets wrap up the tutorial. The methods used above combine the results of different fits and estimate a global 
value of `centre` alongside estimates of its error. 

In this tutorial, we fitted just 5 datasets. Of course, we could easily fit more datasets, and we would find that
as we added more datasets our estimate of the global centre would become more precise.

In the next tutorial, we will compare this result to one inferred via a graphical model. 
"""
