"""
Tutorial 4: Non-linear Search
=============================

Its finally time to take our model and fit it to data.

So, how do we infer the parameters for the 1D `Gaussian` that give a good fit to our data?  Previously, we tried a very
basic approach, randomly guessing models until we found one that gave a good fit and high log_likelihood. We discussed
that this was not a viable strategy for more complex models. Surprisingly, this is the basis of how model fitting
actually works!

Basically, our model-fitting algorithm guesses lots of models, tracking the log likelihood of these models. As the
algorithm progresses, it begins to guess more models using parameter combinations that gave higher log_likelihood
solutions previously. If a set of parameters provided a good fit to the data previously, a model with similar values
probably will too.

This is called a non-linear search, and is where the notion of a "parameter space" comes in. We are essentially
searching the parameter space defined by the log likelihood function we introduced in the previous tutorial. Why is it
called non-linear? Because this function is non-linear.

Non linear searches are a common tool used by scientists in a wide range of fields. We will use a non-linear search
algorithm called `Emcee`, which for those familiar with statistic inference is a Markov Chain Monte Carlo (MCMC)
method. For now, lets not worry about the details of how Emcee actually works. Instead, just picture that a non-linear
search in **PyAutoFit** operates as follows:

 1) Randomly guess a model and map the parameters via the priors to an instance of the model, in this case
 our `Gaussian`.

 2) Use this model instance to generate model data and compare this model data to the data to compute a log likelihood.

 3) Repeat this many times, choosing models whose parameter values are near those of models which have higher log
 likelihood values. If a new model's log likelihood is higher than previous models, new models will be chosen with
 parameters nearer this model.

The idea is that if we keep guessing models with higher log-likelihood values, we will inevitably `climb` up the
gradient of the log likelihood in parameter space until we eventually hit the highest log likelihood models.

To be clear, this overly simplified description of an MCMC algorithm is not how `Emcee` actually works in detail. We
are omitting crucial details on how our priors impact our inference as well as how the MCMC algorithm provides us with
reliable errors on our parameter estimates. The goal of this chapter to teach you how to use **PyAutoFit**, not the
actual details of Bayesian inference. If you are interested in the details of how MCMC works, I recommend you checkout
the following web links:

https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo

https://twiecki.io/blog/2015/11/10/mcmc-sampling/

https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import autofit as af
import autofit.plot as aplt

"""
__Data__

Load the dataset from the `autofit_workspace/dataset` folder.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets remind ourselves what the data looks like, using defining a `plot_profile_1d` method for convenience.

Note that this function has tools for outputting the images to hard-disk as `.png` files, which we'll use later
in this tutorial.
"""


def plot_profile_1d(
    xvalues: np.ndarray,
    profile_1d: np.ndarray,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    errors: Optional[np.ndarray] = None,
    color: Optional[str] = "k",
    output_path: Optional[str] = None,
    output_filename: Optional[str] = None,
):
    """
    Plot a 1D image of data on a plot of x versus y, where the x-axis is the x coordinate of the 1D profile
    and the y-axis is the value of the 1D profile at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues
        The x-coordinates the profile is defined on.
    profile_1d
        The normalization values of the profile which are plotted.
    ylabel
        The y-label of the plot.
    errors
        The errors on each data point, which are related to its noise-map.
    output_path
        The path the image is to be output to hard-disk as a .png.
    output_filename
        The filename of the file if it is output as a .png.
    output_format
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    """
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
    if output_filename is None:
        plt.show()
    else:
        if not path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


xvalues = np.arange(data.shape[0])

plot_profile_1d(
    xvalues=xvalues, profile_1d=data, errors=noise_map, title="Data", ylabel="Data"
)

"""
__Model__

Lets import the `Gaussian` class for this tutorial, which is the model we will fit using the non-linear search.

As discussed in the previous tutorial, by import this from a module rather than defining it in the tutorial script
itself its priors are automatically loaded from configuration files.
"""
import gaussian as g

"""
__Analysis__

The non-linear search requires an `Analysis` class, which:

 - Receives the data to be fitted and prepares it so the model can fit it.
 
 - Defines the `log_likelihood_function` used to compute the `log_likelihood` from a model instance. 
 
 - Passes this `log_likelihood` to the non-linear search so that it can determine parameter values for the next 
 model that it samples.

For our 1D `Gaussian` model-fitting example, here is our `Analysis` class (read the comment in 
the `log_likelihood_function` for a description of how model mapping is used to set up the model that each iteration
of the non-linear search fits):
"""


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `instance` that comes into this method is an instance of the `Gaussian` class above, with the parameters
        set to values chosen by the non-linear search (These are commented out to prevent excessive print statements
        when we run the non-linear search).

        This instance`s parameter values are chosen by the non-linear search based on the priors of each parameter and
        the previous models with the highest likelihood result. They are set up as physical values, by mapping unit
        values chosen by the non-linear search.

        print("Gaussian Instance:")
        print("Centre = ", instance.centre)
        print("Normalization = ", instance.normalization)
        print("Sigma = ", instance.sigma)

        Below, we fit the data with the `Gaussian` instance, using its "model_data_1d_via_xvalues_from" function to create the
        model data.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


"""
__Visualization__

The `Analysis` class above is all we need to fit our model to data with a non-linear search. However, it will provide
us with limited output to inspect whether the fit was a success or not

By extending the `Analysis` class with a `visualize` function, we can perform on-the-fly visualization, which outputs
images of the quantities we described in tutorial 2 to hard-disk as `.png` files using the `plot_profile_1d` function above. 

Visualization of the results of the search, such as the corner plot of what is called the "Probability Density 
Function", are also automatically output during the model-fit on the fly.
"""


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the previous tutorial.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search. The `instance` passed
        into the visualize method is maximum log likelihood solution obtained by the model-fit so far and it can be
        used to provide on-the-fly images showing how the model-fit is going.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0

        """
        The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
        """
        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )


"""
__Search__

To perform the non-linear search using `Emcee`, we simply compose our model using a `Model`, instantiate the 
`Analysis` class and pass them to an instance of the `Emcee` class. 

We also pass a `name` and `path_prefrix`, which specifies that when the results are output to the folder 
`autofit_workspace/output` they'll also be written to the folder `howtofit/chapter_1/tutorial_4_non_linear_search`.
"""
model = af.Model(g.Gaussian)

analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee()

"""
__Model Fit__

We begin the non-linear search by calling its `fit` method. This will take a minute or so to run (which is very fast 
for a model-fit). 
"""
print(
    """
    Emcee has begun running.
    This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

"""
__Result__

Once completed, the non-linear search returns a `Result` object, which contains lots of information about the 
NonLinearSearch.
 
A full description of the `Results` object will be given in tutorial 6 and can also be found at:
 
`autofit_workspace/overview/simple/results`
`autofit_workspace/overview/complex/results`.

The `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
Lets use the `result` it to inspect the maximum likelihood model instance.
"""
print("Maximum Likelihood Model:\n")
max_log_likelihood_instance = result.samples.max_log_likelihood()
print("Centre = ", max_log_likelihood_instance.centre)
print("Normalization = ", max_log_likelihood_instance.normalization)
print("Sigma = ", max_log_likelihood_instance.sigma)

"""
We can use this to plot the maximum log likelihood fit over the data:
"""
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Emcee model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Samples__

Above, we used the `Result`'s `samples` property, which in this case is a `SamplesMCMC` object:
"""
print(result.samples)

"""
This object acts as an interface between the `Emcee` output results on your hard-disk and this Python code. For
example, we can use it to get the parameters and log likelihood of an accepted emcee sample.
"""
print(result.samples.parameter_lists[10][:])
print(result.samples.log_likelihood_list[10])

"""
We can also use it to get a model instance of the `median_pdf` model, which is the model where each parameter is
the value estimated from the probability distribution of parameter space.
"""
mp_instance = result.samples.median_pdf()
print()
print("Median PDF Model:\n")
print("Centre = ", mp_instance.centre)
print("Normalization = ", mp_instance.normalization)
print("Sigma = ", mp_instance.sigma)

"""
The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization 
tool `corner.py`, which is wrapped via the `EmceePlotter` object.

The PDF shows the 1D and 2D probabilities estimated for every parameter after the model-fit. The two dimensional 
figures can show the degeneracies between different parameters, for example how increasing $\sigma$ and decreasing 
the normalization $I$ can lead to similar likelihoods and probabilities.
"""
search_plotter = aplt.EmceePlotter(samples=result.samples)
search_plotter.corner()

"""
The PDF figure above can be seen to have labels for all parameters, whereby sigma appears as a sigma symbol, the
normalization is `N`, and centre is `x`. This is set via the config file `config/notation/label.yaml`. When you write
your own model-fitting code with **PyAutoFit**, you can update this config file so your PDF's automatically have the
correct labels.

we'll come back to the `Samples` objects in tutorial 6!

__Output to Hard-Disk__

The non-linear search `dynesty` above did not output results to hard-disk, which for quick model-fits and
experimenting with different models is desirable.

For many problems it is preferable for all results to be written to hard-disk. The benefits of doing this include:

- Inspecting results in an ordered directory structure can be more efficient than using a Jupyter Notebook.
- Results can be output on-the-fly, to check that a fit is progressing as expected mid way through.
- An unfinished run can be resumed where it was terminated.
- Additional information about a fit (e.g. visualization) can be output.
- On high performance computers which use a batch system, this is the only way to transfer results.

Any model-fit performed by **PyAutoFit** can be saved to hard-disk, by simply giving the non-linear search a 
`name`. A `path_prefix` can optionally be input to customize the output directory.

__Unique Identifier__

In the output folder, you will note that results are in a folder which is a collection of random characters. This acts 
as a `unique_identifier` of the model-fit, where this identifier is generated based on the model, priors and search that 
are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, priors or search,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder. 

__Contents__

In particular, you'll find (in a folder that is a random string of characters):

 - `model.info`: A file listing every model component, parameter and prior in your model-fit.

 - `model.results`: A file giving the latest best-fit model, parameter estimates and errors of the fit.
 
 - `output.log`: A file containing the text output of the non-linear search.
 
 - `samples`: A folder containing the `Emcee` output in hdf5 format.txt (you'll probably never need to look at these, 
   but its good to know what they are).
 
 - `search.summary` A file containing information on the search, including the total number of samples,
 overall run-time and time it takes to evaluate the log likelihood function.
 
 - `image`: A folder containing `.png` files of the fits defined in the `visualize` method.
 
 - Other metadata which you can ignore for now (e.g. the pickles folder).
"""
search = af.Emcee(
    path_prefix=path.join("howtofit", "chapter_1"), name="tutorial_4_non_linear_search"
)

print(
    """
    Emcee has begun running - checkout the autofit_workspace/output/howtofit/tutorial_4_non_linear_search
    folder for live output of the results.
    This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")
