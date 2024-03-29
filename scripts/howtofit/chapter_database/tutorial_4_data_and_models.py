"""
Tutorial 4: Data and Models
===========================

Up to now, we've used used the `Aggregator` to load and inspect the `Result` and `Samples` of 3 model-fits.

In this tutorial, we'll look at how write Python generators which use the `Aggregator` to inspect, interpret and plot
the results of the model-fit, including fitting and plotting different models to our data.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af

"""
__Database File__

We begin by loading the database via the `.sqlite` file as we did in the previous tutorial. 
"""
database_file = "database_howtofit.sqlite"
agg = af.Aggregator.from_database(filename=database_file, completed_only=True)

"""
__Plot Function__

We'll reuse the `plot_profile_1d` function of previous tutorials, however it now displays to the notebook as opposed to
outputting the results to a .png file.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_profile_1d(
    xvalues, profile_1d, title=None, ylabel=None, errors=None, color="k"
):
    plt.errorbar(
        x=xvalues,
        y=profile_1d,
        yerr=errors,
        color=color,
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()


"""
__Dataset Loading__

We can use the `Aggregator` to load a generator of every fit`s data, by changing the `output` attribute to the 
`data` attribute at the end of the aggregator.

Note that in the `Analysis` class of tutorial 1, we specified that the `data` object would be saved to hard-disc using
the `save_attributes` method, so that the `Aggregator` can load it.
"""
print(len(agg))
data_gen = agg.values("data")
print("Datas:")
print(list(data_gen), "\n")

"""
We can plot the `data` using the `plot_profile_1d` method.
"""
for data in agg.values("data"):
    plot_profile_1d(
        xvalues=np.arange(data.shape[0]),
        profile_1d=data,
        title="Data",
        ylabel="Data Values",
        color="k",
    )

"""
We can repeat the same trick to get the `noise_map` of every fit.
"""
noise_map_gen = agg.values("noise_map")
print("Noise-Maps:")
print(list(noise_map_gen), "\n")

"""
The `info` dictionary we input into the `NonLinearSearch` is also available.
"""
for info in agg.values("info"):
    print(info)

"""
__Fitting via Lists__

We're going to refit each dataset with the `max_log_likelihood_instance` of each model-fit and plot the residuals.

(If you are unsure what the `zip` is doing below, it essentially combines the `data_gen`, `noise_map_gen` and
`samples_gen` into one list such that we can iterate over them simultaneously).
"""
samples_gen = agg.values("samples")
data_gen = agg.values("data")
noise_map_gen = agg.values("noise_map")

for data, noise_map, samples in zip(data_gen, noise_map_gen, samples_gen):
    instance = samples.max_log_likelihood()

    xvalues = np.arange(data.shape[0])

    model_data = sum(
        [
            profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
            for profile in instance
        ]
    )

    residual_map = data - model_data

    plot_profile_1d(
        xvalues=xvalues,
        profile_1d=residual_map,
        title="Residual Map",
        ylabel="Residuals",
        color="k",
    )

"""
__Fitting via Generators__

There is a problem with how we plotted the residuals above, can you guess what it is?

We used lists! If we had fit a large sample of data, the above object would store the data of all objects 
simultaneously in memory on our hard-disk, likely crashing our laptop! To avoid this, we must write functions that 
manipulate the `Aggregator` generators as generators themselves. Below is an example function that performs the same 
task as above.
"""


def plot_residuals_from(fit):
    data = fit.value(name="dataset.data")
    noise_map = fit.value(name="dataset.noise_map")

    xvalues = np.arange(data.shape[0])

    model_data = sum(
        [
            profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
            for profile in fit.instance
        ]
    )

    residual_map = data - model_data

    plot_profile_1d(
        xvalues=xvalues,
        profile_1d=residual_map,
        title="Residual Map",
        ylabel="Residuals",
        color="k",
    )


"""
To manipulate this function as a generator using the `Aggregator`, we apply it to the `Aggregator`'s `map` function.
"""
plot_residuals_gen = agg.map(func=plot_residuals_from)

"""
Lets get the `max_log_likelihood_instance`s, as we did in tutorial 3.
"""
instances = [samps.max_log_likelihood() for samps in agg.values("samples")]

"""
Okay, we want to inspect the fit of each `max_log_likelihood_instance`. To do this, we reperform each fit.

First, we need to create the `model_data` of every `max_log_likelihood_instance`. Lets begin by creating a list 
of profiles of every model-fit.
"""
