"""
Tutorial 2: Fitting Data
========================

We have now learnt that a model is a set of equations, numerical processes and assumptions describing
a physical system. We defined a couple of simple models made of 1D equations like a Gaussian, composed them as models
in **PyAutoFit** using the `Model` and `Collection` objects, and used these models to create model data for different
values of their parameters.

For our model to inform us about a real physical system, we need to fit it to data. By fitting it to the data,
we can determine whether the model provides a good or bad fit to the data. If it is a good fit, we will learn which
parameter values best describe the data and therefore the physical system as a whole. If it is a bad fit, we will
learn that our model is not representative of the physical system and therefore that we need to change it.

The process of defining a model, fitting it to data and using it to learn about the system we are modeling is at the
heart of model-fitting. One would typically repeat this process many times, making the model more complex to better
fit more data, better describing the physical system we are interested in.

In Astronomy, this is the process that was followed to learn about the distributions of stars in galaxies. Fitting
high quality images of galaxies with ever more complex models, allowed astronomers to determine that the stars in
galaxies are distributed in structures like disks, bars and bulges, and it taught them that stars appear differently
in red and blue images due to their age.

In this tutorial, we will learn how to fit the `model_data` created by a model to data, and we will in particular:

 - Load data of a 1D Gaussian signal which is the data we will fit.

 - Subtract the model data from the data to compute quantities like the residuals of the fit.

 - Quantify the goodness-of-fit of a model to the data quantitatively using a key quantity in model-fitting called the
   `log_likelihood`.

This will all be performed using the **PyAutoFit** API for model composition, which forms the basis of all model
fitting performed by **PyAutoFit**.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from os import path
import matplotlib.pyplot as plt
import numpy as np

"""
__Data__

Our data is noisy 1D data containing a signal, where the underlying signal is generated using the equation of 
a 1D Gaussian, a 1D Exponential or a sum of multiple 1D profiles.
 
We now load this data from .json files, where:

 - The `data` is a 1D numpy array of values corresponding to the observed signal.
 - The `noise_map` is a 1D numpy array of values corresponding to the estimate noise value in every data point.
 
These datasets are created via the scripts `autofit_workspace/howtofit/simulators`, feel free to check them out!
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))

noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
We now plot the 1D signal via `matplotlib`.

The 1D signal is observed on uniformly spaced `xvalues`, which are computed using the `arange` function 
and `data.shape[0]` method.

These x values will be used again below, when we create model data from the model.
"""
xvalues = np.arange(data.shape[0])
plt.plot(xvalues, data, color="k")
plt.title("1D Dataset Containing a Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Signal Value")
plt.show()

"""
The plot above only showed the signal, and did not show the noise estimated in every data point. 

We can plot the signal, including its `noise_map`, using the `matplotlib` `errorbar` function. 
"""
plt.errorbar(
    xvalues, data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian dataset with errors from the noise-map.")
plt.xlabel("x values of profile")
plt.ylabel("Signal Value")
plt.show()


"""
__Model Data__

How do we actually fit our `Gaussian` model to this data? First, we generate `model_data` of the 1D `Gaussian` model,
following the same steps as the previous tutorial. 
"""


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian`s model parameters.
        sigma=5.0,
    ):
        """
        Represents a 1D Gaussian profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to fit example datasets
        via a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray):
        """

        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


"""
We can use this model to create `model_data` of the `Gaussian` by passing it an input `xvalues` of the observed
data.

We do this below, and plot the resulting model-data.
"""
model = af.Model(Gaussian)

gaussian = model.instance_from_vector(vector=[60.0, 20.0, 15.0])

model_data = gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)

plt.plot(xvalues, model_data, color="r")
plt.title("1D Gaussian model.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
It is often more informative to plot the `data` and `model_data` on the same plot for comparison.
"""
plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Model-data fit to 1D Gaussian data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
Different values of `centre`, `normalization` and `sigma` change the `Gaussian``s appearance. 

Have a go at editing some of the values input into `instance_from_vector()`, recomputing the `model_data` and
plotting it above to see this behaviour.

__Residuals__

The comparison of the `data` and `model_data` above is informative, but it can be more useful to show the
residuals, which are calculated as `data - model_data` in 1D:
"""
residual_map = data - model_data
plt.plot(xvalues, residual_map, color="r")
plt.title("Residuals of model-data fit to 1D Gaussian data.")
plt.xlabel("x values of profile")
plt.ylabel("Residuals")
plt.show()
plt.clf()

"""
Are these residuals a good fit to the data? Without considering the noise in the data, we can't be sure.

We can plot the residual-map with error-bars for the noise-map, which below shows that the model is a pretty bad fit,
because many of the residuals are far away from 0 even after accounting for the noise in every data point.
"""
residual_map = data - model_data
plt.errorbar(
    x=xvalues,
    y=residual_map,
    yerr=noise_map,
    color="r",
    ecolor="r",
    elinewidth=1,
    capsize=2,
)
plt.title("Residuals of model-data fit to 1D Gaussian data.")
plt.xlabel("x values of profile")
plt.ylabel("Residuals")
plt.show()
plt.clf()

"""
__Normalized Residuals__

A different way to quantify and visualize how good (or bad) the fit is, is using the normalized residual-map (sometimes
called the standardized residuals).

This is defined as the residual-map divided by the noise-map. 

If you are familiar with the concept of `sigma` variancdes in statistics, the normalized residual-map is equivalent
to the number of `sigma` the residual is from zero. For example, a normalized residual of 2.0 (which has confidence
internals for 95%) means that the probability that the model under-estimates the data by that value is just 5.0%.

The residual map with error bars and normalized residual map portray the same information, but the normalized
residual map is better for visualization for problems with more than 1 dimension, as plotting the error bars in
2D or more dimensions is not straight forward.
"""
normalized_residual_map = residual_map / noise_map
plt.plot(xvalues, normalized_residual_map, color="r")
plt.title("Normalized residuals of model-data fit to 1D Gaussian data.")
plt.xlabel("x values of profile")
plt.ylabel("Normalized Residuals")
plt.show()
plt.clf()

"""
__Chi Squared__

We now define the `chi_squared_map`, which is the `normalized_residual_map` squared, and will be used to compute the
the final goodness of fit measure.

The normalized residual map has both positive and negative values. When we square it, we therefore get only positive
values. This means that a normalized residual of -0.2 and 0.2 both become 0.04, and therefore in the context of a
`chi_squared` signify the same goodness-of-fit.

Again, it is clear that the model gives a poor fit to the data.
"""
chi_squared_map = (normalized_residual_map) ** 2
plt.plot(xvalues, chi_squared_map, color="r")
plt.title("Chi-Squared Map of model-data fit to 1D Gaussian data.")
plt.xlabel("x values of profile")
plt.ylabel("Chi-Squareds")
plt.show()
plt.clf()

"""
we now reduce all the information in our `chi_squared_map` into a single goodness-of-fit measure by defining the 
`chi_squared`: the sum of all values in the `chi_squared_map`.

This is why having all positive and negative values in the normalized residual map become positive is important,
as this summed measure would otherwise cancel out the positive and negative values.
"""
chi_squared = np.sum(chi_squared_map)
print("Chi-squared = ", chi_squared)

"""
The lower the chi-squared, the fewer residuals in the model's fit to the data and therefore the better our fit as
a whole!

__Noise Normalization__

We now define a second quantity that will enter our final quantification of the goodness-of-fit, called the
`noise_normalization`.

This is the log sum of all noise values squared in our data. Given the noise-map is fixed, the `noise_normalization`
retains the same value for all models that we fit, and therefore could be omitted. Nevertheless, its good practise
to include it as it has an important meaning statistically.

Lets not worry about what a `noise_normalization` actually means, because its not important for us to successfully
get a model to fit a dataset. In a nutshell, it relates the noise in the dataset being drawn from a Gaussian
distribution.
"""
noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))

"""
__Likelihood__

From the `chi_squared` and `noise_normalization` we can define a final goodness-of-fit measure, the `log_likelihood`. 

This is the sum of the `chi_squared` and `noise_normalization` multiplied by -0.5. Why -0.5? Again, lets not worry
about this for now.
"""
log_likelihood = -0.5 * (chi_squared + noise_normalization)
print("Log Likelihood = ", log_likelihood)

"""
Above, we stated that a lower `chi_squared` corresponds to a better model-data fit. 

When computing the `log_likelihood` we multiplied the `chi_squared` by -0.5. Therefore, a higher log likelihood
corresponds to a better model fit, as one would hope!

__Fitting Functions__

If you are familiar with model-fitting, you'll have probably heard of terms like 'residuals', 'chi-squared' and
'log_likelihood' before. These are the standard metrics by which a model-fit`s quality is quantified. They are used for
model fitting in general, so not just when your data is 1D but when its a 2D image, 3D datacube or something else
entirely!

If you have not performed model fitting before and these terms are new to you, make sure you are clear on exactly what
they all mean as they are at the core of all model-fitting performed in **PyAutoFit** (and statistical inference in
general)!

Lets recap everything we've learnt so far:
    
 - We can define a model, like a 1D `Gaussian`, using Python classes that follow a certain format.
 
 - The model can be set up as a `Collection` and `Model`, having its parameters mapped to an instance of the
   model class (e.g the `Gaussian`).  

 - Using this model instance, we can create model-data and compare it to data and quantify the goodness-of-fit via a 
   log likelihood.

We now have everything we need to fit our model to our data! 

So, how do we go about finding the best-fit model? That is, what model which maximizes the log likelihood?

The most simple thing we can do is guess parameters. When we guess parameters that give a good fit (e.g. a higher 
log likelihood), we then guess new parameters with values near those previous vlaues. We can repeat this process, 
over and over, until we find a really good model!

For a 1D  `Gaussian` this works pretty well. Below, we fit 3 different `Gaussian` models and end up landing on
the best-fit model (the model I used to create the dataset in the first place!).

For convenience, I've create functions which compute the `log_likelihood` of a model-fit and plot the data and model
data with errors.
"""


def log_likelihood_from(data, noise_map, model_data):
    residual_map = data - model_data
    normalized_residual_map = residual_map / noise_map
    chi_squared_map = (normalized_residual_map) ** 2
    chi_squared = sum(chi_squared_map)
    noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
    log_likelihood = -0.5 * (chi_squared + noise_normalization)

    return log_likelihood


def plot_model_fit(xvalues, data, noise_map, model_data, color="k"):
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color=color,
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.plot(xvalues, model_data, color="r")
    plt.title("Fit of model-data to data.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile Value")
    plt.show()
    plt.clf()


"""
__Guess 1__
"""

gaussian = model.instance_from_vector(vector=[50.0, 10.0, 5.0])
model_data = gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)
plot_model_fit(
    xvalues=xvalues,
    data=data,
    noise_map=noise_map,
    model_data=model_data,
    color="r",
)

log_likelihood = log_likelihood_from(
    data=data, noise_map=noise_map, model_data=model_data
)
print(f"Log Likelihood: {log_likelihood}")

"""
__Guess 2__
"""

gaussian = model.instance_from_vector(vector=[50.0, 25.0, 5.0])
model_data = gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)
plot_model_fit(
    xvalues=xvalues,
    data=data,
    noise_map=noise_map,
    model_data=model_data,
    color="r",
)

log_likelihood = log_likelihood_from(
    data=data, noise_map=noise_map, model_data=model_data
)
print(f"Log Likelihood: {log_likelihood}")

"""
__Guess 3__
"""

gaussian = model.instance_from_vector(vector=[50.0, 25.0, 10.0])
model_data = gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)
plot_model_fit(
    xvalues=xvalues,
    data=data,
    noise_map=noise_map,
    model_data=model_data,
    color="r",
)

log_likelihood = log_likelihood_from(
    data=data, noise_map=noise_map, model_data=model_data
)
print(f"Log Likelihood: {log_likelihood}")

"""
__Extensibility__

Fitting models made of multiple components is straight forward. 

We again simply create the model via  the `Collection` object, use it to generate `model_data` and fit it to the 
data in order to compute the log likelihood.
"""
model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

instance = model.instance_from_vector(vector=[40.0, 0.2, 0.3, 60.0, 0.5, 1.0])

model_data_0 = instance.gaussian_0.model_data_1d_via_xvalues_from(xvalues=xvalues)
model_data_1 = instance.gaussian_1.model_data_1d_via_xvalues_from(xvalues=xvalues)

model_data = model_data_0 + model_data_1

"""
We plot the data and model data below, showing that we get a bad fit (a low log likelihood) for this model.
"""
plot_model_fit(
    xvalues=xvalues,
    data=data,
    noise_map=noise_map,
    model_data=model_data,
    color="r",
)

log_likelihood = log_likelihood_from(
    data=data, noise_map=noise_map, model_data=model_data
)
print(f"Log Likelihood: {log_likelihood}")


"""
When the model had just 3 parameters, it was feasible to guess values by eye and find a good fit. 

With six parameters, this approach becomes inefficient, and doing it with even more parameters would be impossible!

In the next turorial, we will learn a more efficient and automated approach for fitting models to data.

__Wrap Up__

To end, have another quick think about the model you ultimately want to fit with **PyAutoFit**. What does the
data look like? Is it one dimension? two dimensions? Can you easily define a model which generates realizations of
this data? Can you picture what a residual map would look like and how you would infer a log likelihood from it?

If not, don't worry about it for now, because you first need to learn how to fit a model to data using **PyAutoFit**.
"""
