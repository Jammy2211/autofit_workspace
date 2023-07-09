"""
Tutorial 3: Non Linear Search
=============================

In the previous tutorials, we defined a model and fitted it to data via fitting functions. We quantified the goodness
of fit via the log likliehood and showed that for models with only a few free parameters, we could find good fits to
the data by manually guessing parameter values. However, for more complex models, this approach was infeasible.

In this tutorial, we will learn how to fit the model to data properly, using a technique that can scale up to
models with 10s or 100s of parameters.

__Parameter Space__

In mathematics, we can write a function as follows:

$f(x) = x^2$

In this function, when we input the parameter $x$ in to the function $f$, it returns a value $f(x)$.

This mapping between values of $x$ and $f(x)$ define the "parameter space" of this function (which fot
the function $f(x) = x^2$ is a parabola).

A function can have multiple parameters, for 3 parameters, $x$, $y$ and $z$:

$f(x, y, z) = x + y^2 - z^3$

The mapping between values of $x$, $y$ and $z$ and $f(x, y, z)$ define another parameter space, albeit it now
has 3 dimensions.

The concept of a parameter space relates closely to how in the previous tutorial we use instances of a 1D Gaussian
profile, with parameters $(x, I, \sigma)$ to fit data with a model and compute a log likelihood.

This process can be thought of as a function $f (x, I, \sigma)$, where the value returned by this function is the
log likelihood.

With that, we have introduced one of the most important concepts in model-fitting,
the "log likelihood function". This function describes how we use an instance of the model (e.g. where the
parameters have values) to compute a log likelihood describing good of a fit to the data it is.

We can write this log likelihood function as follows:

$f(x, N, \sigma) = log_likelihood$

By expressing the likelihood in this way, we can therefore now think of our model as having a parameter space. This
parameter space consists of an N dimensional surface (where N is the number of free parameters) spanning all possible
values of model parameters. This surface itself can be considered the "likelihood surface", and finding the peak of
this surface is our goal when we perform model-fitting.

This parameter space is "non-linear", meaning that the relationship between input parameters and log likelihood does
not behave linearly. This simply means that it is not possible to predict what a log likelihood will be from a set of
model parameters, unless a whole fit to the data is performed in order to compute the value.

__Non Linear Search__

Now that we are thinking about the problem in terms of a non-linear parameter space with a likelihood surface, we can
now introduce the method used to fit the model to the data, the "non-linear search".

Previously, we tried a basic approach, randomly guessing models until we found one that gave a good fit and
high `log_likelihood`. Surprisingly, this is the basis of how model fitting using a non-linear search actually works!

The non-linear search guesses lots of models, tracking the log likelihood of these models. As the algorithm
progresses, it preferentially tries more models using parameter combinations that gave higher log likelihood solutions
previously. The rationale is that if a parameters set provided a good fit to the data, models with similar values will
too.

There are two key differences between guessing random models to find a good fit and a non-linear search:

 - The non-linear search fits the model to the data in mere miliseconds. It therefore can compute the log likelihood
   of tens of thousands of different model parameter combinations in order to find the highest likelihood solutions.
   This would have been impractical for a human.

 - The non-linear search has a much better tracking system to remember which models it guess previously and what
   their log likelihoods were. This means it can sample all possible solutions more thoroughly, whilst honing in on
   those which give the highest likelihood more quickly.

We can think of our non-linear search as "searching" parameter space, trying to find the regions of parameter space
with the highest log likelihood values. Its goal is to find them, and then converge on the highest log likelihood
solutions possible. In doing so, it can tell us what model parameters best-fit the data.

This picture of how a non-linear search is massively simplified, and omits key details on how statistical principles
are upheld to ensure that results are statistically robust. The goal of this chapter is to teach you how to fit a
model to data, not the underlying principles of Bayesian inference on which model-fitting is based.

If you are interested, more infrmation can be found at the following web links:

https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo

https://twiecki.io/blog/2015/11/10/mcmc-sampling/

https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50

__MCMC__

There are many different non-linear search algorithms, which search parameter space in different ways. This tutorial
uses a a Markov Chain Monte Carlo (MCMC) method alled `Emcee`. For now, lets not worry about the details of how
an MCMC method actually works, and just use the simplified picture we painted above.
"""
import autofit as af
import autofit.plot as aplt
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

"""
__Data__

Load and plot the dataset from the `autofit_workspace/dataset` folder.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

xvalues = np.arange(data.shape[0])
print(xvalues)

plt.errorbar(
    xvalues, data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
__Model__

Lets again define our 1D `Gaussian` model. 
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
We now compose our model, a single 1D Gaussian, which we will fit to the data via the non-linear search.
"""
model = af.Model(Gaussian)

print(model.info)

"""
__Priors__

When we print its `.info`, we see that the parameters have priors (e.g. `UniformPrior`). We have so far not worried 
about what these meant, but now we understand how a non-linear search works we can now discuss what priors are.

A parameter, for example the `centre` of the `Gaussian`, could take any value between negative and positive infinity. 
However, when we inspect the data it is clearly confined to values between 0.0 and 100.0. Our model parameter space 
should reflect this, and only contain solutions with these physically plausible values between 0.0 --> 100.0.

One role of priors is to define where parameter space has valid solutions. The `centre` parameter has 
a `UniformPrior` with a  `lower_limit=0.0` and `upper_limit=100.0`. It therefore is already confined to the values 
discussed above.

Priors have a second role: they encode our previous beliefs about a model and what values we expect the parameters 
to have. 

For example, imagine we had multiple datasets observing the same signal and we had already fitted the model to the 
first signal already. We may set priors that reflect this result, as we have prior knowledge of what the parameters
will likely be. 

Setting priros in this way actually changes the result inferred when fitting the second dataset, because the priors 
partly constrain the result based on the information learned in the first fit. Other types of priors you will 
see throughout the autofit workspace (e.g `GaussianPrior`, `LogUniformPrior`) allow one to encode this type of 
information in a fit..

In this tutorial, we will stick to uniform priors, as they are conceptually the most simple.

Lets manually set the priors of the model we fit in this example.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
__Analysis__

The non-linear search requires an `Analysis` class, which:

 1) Receives the data that the model fits.

 2) Defines the `log_likelihood_function`, which computes a `log_likelihood` from a model instance. 

 3) Provides an interface between the non-linear search and the `log_likelihood_function`, so the search can determine
    the goodness of fit of any set of model parameters.
    
The non-linear search calls the `log_likelihood_function` many times, enabling it map out the high likelihood regions 
of parameter space and converges on the highest log likelihood solutions.

Below is a suitable `Analysis` class for fitting a 1D gaussian to the data loaded above.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

        Its `log_likelihood_function` defines how the model is fitted to the data and it is called many times by
        the non-linear search fitting algorithm.

        In this example the `Analysis` `__init__` constructor only contains the `data` and `noise-map`, but it can be
        easily extended to include other quantities.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric, the log likelihood.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

        The `instance` that comes into this method is an instance of the `Gaussian` model above. The parameter values
        are chosen by the non-linear search, based on where it thinks the high likelihood regions of parameter
        space are.

        The lines of Python code are commented out below to prevent excessive print statements when we run the
        non-linear search, but feel free to uncomment them and run the search to see the parameters of every instance
        that it fits.

        print("Gaussian Instance:")
        print("Centre = ", instance.centre)
        print("Normalization = ", instance.normalization)
        print("Sigma = ", instance.sigma)

        The data is fitted using an `instance` of the `Gaussian` class where its `model_data_1d_via_xvalues_from`
        is called in order to create a model data representation of the Gaussian that is fitted to the data.
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
We create an instance of the `Analysis` class by simply passing it the `data` and `noise_map`:
"""
analysis = Analysis(data=data, noise_map=noise_map)

"""
__Search__

To use the non-linear search `Emcee` we simply create an instance of the `af.Emcee` object and pass the analysis
and model to its `fit` method.
"""
model = af.Model(Gaussian)

search = af.Emcee()

"""
__Model Fit__

We begin the non-linear search by calling its `fit` method. This will take a minute or so to run.
"""
print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

Upon completion the non-linear search returns a `Result` object, which contains information about the model-fit.

The `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
The result has a "maximum log likelihood instance", which is the instance of the model (e.g. the `Gaussian`) with
the model parameters that gave the highest overall log likelihood out of any model trialed by the non-linear search.
"""
print("Maximum Likelihood Model:\n")
max_log_likelihood_instance = result.samples.max_log_likelihood()
print("Centre = ", max_log_likelihood_instance.centre)
print("Normalization = ", max_log_likelihood_instance.normalization)
print("Sigma = ", max_log_likelihood_instance.sigma)

"""
We can use this to plot the maximum log likelihood fit over the data and confirm that a good fit was inferred:
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
The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization 
tool `corner.py`, which is wrapped via the `EmceePlotter` object.

The PDF shows the 1D and 2D probabilities estimated for every parameter after the model-fit. The two dimensional 
figures can show the degeneracies between different parameters, for example how increasing $\sigma$ and decreasing 
the normalization $I$ can lead to similar likelihoods and probabilities.
"""
search_plotter = aplt.EmceePlotter(samples=result.samples)
search_plotter.corner()

"""
A more detailed description of the `Result` object will be given in tutorial 5.

__Wrap Up__

This tutorial introduced a lot of concepts: the parameter space, likelihood surface, non-linear search, priors, 
and much more. 

Make sure you are confident in your understanding of them, however the next tutorial will expand on them all.
"""
