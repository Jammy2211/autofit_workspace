"""
Overview: The Basics
--------------------

PyAutoFit is a Python based probabilistic programming language which enables detailed Bayesian analysis of scientific
datasets, with everything necessary to scale-up Bayesian analysis to complex models and big datasets.

In this overview, we introduce the basic API for model-fitting with **PyAutoFit**, including how to define a likelihood
function, compose a probabilistic model, and fit it to data via a non-linear fitting algorithm (e.g.
Markov Chain Monta Carlo (MCMC), maximum likelihood estimator, nested sampling).

If you have previous experience performing model-fitting and Bayesian inference this will all be very familiar, but
we'll highlight some benefits of using **PyAutoFit** instead of setting up the modeling manually yourself (e.g. by
wrapping an MCMC library with your likelihood function).

The biggest benefits of using **PyAutoFit** are presented after we've introduced the API and can be summarized as follows:

- **The scientific workflow**: streamline detailed modeling and analysis of small datasets with tools enabling the scaling up to large datasets.

- **Statistical Inference Methods**: Dedicated functionality for many advanced statical inference methods, including Bayesian hierarchical analysis, model comparison and graphical models.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt

import matplotlib.pyplot as plt
import numpy as np
import os
from os import path

"""
__Example Use Case__

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a noisy 1D signal. 

We load the example 1D data containing this noisy signal below, which is included with the `autofit_workspace`
in .json files.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
We plot the data containing the noisy 1D signal.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("Example Data")
plt.xlabel("x values of data (pixels)")
plt.ylabel("Signal Value")
plt.show()
plt.close()

"""
The 1D signal was generated using a 1D Gaussian profile of the form:

\begin{equation*}
g(x, I, \sigma) = \frac{N}{\sigma\sqrt{2\pi}} \exp{(-0.5 (x / \sigma)^2)}
\end{equation*}

Where:

`x`: Is the x-axis coordinate where the `Gaussian` is evaluated.

`N`: Describes the overall normalization of the Gaussian.

$\sigma$: Describes the size of the Gaussian (Full Width Half Maximum = $\mathrm {FWHM}$ = $2{\sqrt {2\ln 2}}\;\sigma$)

Our modeling task is to fit the signal with a 1D Gaussian and recover its parameters (`x`, `N`, `sigma`).

__Model__

We therefore need to define a 1D Gaussian as a "model component" in **PyAutoFit**.

A model component is written as a Python class using the following format:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor (the `__init__` method) are the parameters of the model, in this case `centre`, `normalization` and `sigma`.
  
- The default values of the input arguments define whether a parameter is a single-valued `float` or a  multi-valued `tuple`. In this case, all 3 input parameters are floats.
  
- It includes functions associated with that model component, which will be used when fitting the model to data.
"""


class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which can be treated as a PyAutoFit
        model-component whose free parameters (centre, normalization and sigma)
        are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray) -> np.ndarray:
        """
        Returns the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the
        Gaussian, by subtracting its centre.

        The output is referred to as the `model_data` to signify that it is
        a representation of the data from the model.

        Parameters
        ----------
        xvalues
            The x coordinates for which the Gaussian is evaluated.
        """
        transformed_xvalues = xvalues - self.centre

        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


"""
To compose a model using the `Gaussian` class above we use the `af.Model` object.
"""
model = af.Model(Gaussian)
print("Model `Gaussian` object: \n")
print(model)

"""
The model has a total of 3 parameters:
"""
print(model.total_free_parameters)

"""
All model information is given by printing its `info` attribute.

This shows that each model parameter has an associated prior.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
The priors can be manually altered as follows, noting that these updated files will be used below when we fit the
model to data.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

"""
Printing the `model.info` displayed these updated priors.
"""
print(model.info)

"""
__Instances__

Instances of the model components above (created via `af.Model`) can be created, where an input `vector` of
parameters is mapped to create an instance of the Python class of the model.

We first need to know the order of parameters in the model, so we know how to define the input `vector`. This
information is contained in the models `paths` attribute:
"""
print(model.paths)

"""
We input values for the 3 free parameters of our model following the order of paths above:
 
 1) `centre=30.0`
 2) `normalization=2.0` 
 3) `sigma=3.0` 
 
This creates an `instance` of the Gaussian class via the model. 
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

"""
This is an instance of the `Gaussian` class.
"""
print("Model Instance: \n")
print(instance)

"""
It has the parameters of the `Gaussian` with the values input above.
"""
print("Instance Parameters \n")
print("x = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
We can use functions associated with the class, specifically the `model_data_1d_via_xvalues_from` function, to 
create a realization of the `Gaussian` and plot it.
"""
xvalues = np.arange(0.0, 100.0, 1.0)

model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

plt.plot(xvalues, model_data, color="r")
plt.title("1D Gaussian Model Data.")
plt.xlabel("x values of profile")
plt.ylabel("Gaussian Value")
plt.show()
plt.clf()

"""
This "model mapping", whereby models map to an instances of their Python classes, is integral to the core **PyAutoFit**
API for model composition and fitting.

__Analysis__

Now we've defined our model, we need to inform **PyAutoFit** how to fit it to data.

We therefore define an `Analysis` class, which includes:

 - An `__init__` constructor, which takes as input the `data` and `noise_map`. This could be extended to include anything else necessary to fit the model to the data.

 - A `log_likelihood_function`, which defines how given an `instance` of the model we fit it to the data and return a log likelihood value.

Read the comments and docstrings of the `Analysis` object below in detail for more insights into how this object
works.
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

        The data is fitted using an `instance` of the `Gaussian` class where its `model_data_1d_via_xvalues_from`
        is called in order to create a model data representation of the Gaussian that is fitted to the data.
        """

        """
        The `instance` that comes into this method is an instance of the `Gaussian` model above, which was created
        via `af.Model()`. 

        The parameter values are chosen by the non-linear search, based on where it thinks the high likelihood regions 
        of parameter space are.

        The lines of Python code are commented out below to prevent excessive print statements when we run the
        non-linear search, but feel free to uncomment them and run the search to see the parameters of every instance
        that it fits.
        """

        # print("Gaussian Instance:")
        # print("Centre = ", instance.centre)
        # print("Normalization = ", instance.normalization)
        # print("Sigma = ", instance.sigma)

        """
        Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
        """
        xvalues = np.arange(self.data.shape[0])

        """
        Use these xvalues to create model data of our Gaussian.
        """
        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        """
        Fit the model gaussian line data to the observed data, computing the residuals, chi-squared and log likelihood.
        """
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


"""
Create an instance of the `Analysis` class by passing the `data` and `noise_map`.
"""
analysis = Analysis(data=data, noise_map=noise_map)

"""
__Non Linear Search__

We have defined the model that we want to fit the data, and the analysis class that performs this fit.

We now choose our fitting algorithm, called the "non-linear search", and fit the model to the data.

For this example, we choose the nested sampling algorithm Dynesty. A wide variety of non-linear searches are 
available in **PyAutoFit** (see ?).
"""
search = af.DynestyStatic(
    nlive=100,
    sample="rwalk",
    number_of_cores=1,
)

"""
__Model Fit__

We begin the non-linear search by calling its `fit` method. 

This will take a minute or so to run.
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

The result object returned by the fit provides information on the results of the non-linear search. 

The `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
Results are returned as instances of the model, as we illustrated above in the model mapping section.

For example, we can print the result's maximum likelihood instance.
"""
print(result.max_log_likelihood_instance)

print("\n Model-fit Max Log-likelihood Parameter Estimates: \n")
print("Centre = ", result.max_log_likelihood_instance.centre)
print("Normalization = ", result.max_log_likelihood_instance.normalization)
print("Sigma = ", result.max_log_likelihood_instance.sigma)

"""
A benefit of the result being an instance is that we can use any of its methods to inspect the results.

Below, we use the maximum likelihood instance to compare the maximum likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Dynesty model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Samples__

The results object also contains a `Samples` object, which contains all information on the non-linear search.

This includes parameter samples, log likelihood values, posterior information and results internal to the specific
algorithm (e.g. the internal dynesty samples).

This is described fully in the results overview, below we use the samples to plot the probability density function
corner of the results.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Extending Models__

The model composition API is designed to  make composing complex models, consisting of multiple components with many 
free parameters, straightforward and scalable.

To illustrate this, we will extend our model to include a second component, representing a symmetric 1D Exponential
profile, and fit it to data generated with both profiles.

Lets begin by loading and plotting this data.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)
xvalues = range(data.shape[0])
plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("Example Data With Multiple Components")
plt.xlabel("x values of data (pixels)")
plt.ylabel("Signal Value")
plt.show()
plt.close()

"""
We define a Python class for the `Exponential` model component, exactly as we did for the `Gaussian` above.
"""


class Exponential:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Exponentials`s model parameters.
        rate=0.01,
    ):
        """
        Represents a symmetric 1D Exponential profile.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        ratw
            The decay rate controlling has fast the Exponential declines.
        """
        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray):
        """
        Returns the symmetric 1D Exponential on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )


"""
We can easily compose a model consisting of 1 `Gaussian` object and 1 `Exponential` object using the `af.Collection`
object:
"""
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

"""
A `Collection` behaves analogous to a `Model`, but it contains a multiple model components.

We can see this by printing its `paths` attribute, where paths to all 6 free parameters via both model components
are shown.

The paths have the entries `.gaussian.` and `.exponential.`, which correspond to the names we input into  
the `af.Collection` above. 
"""
print(model.paths)

"""
We can use the paths to customize the priors of each parameter.
"""
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
All of the information about the model created via the collection can be printed at once using its `info` attribute:
"""
print(model.info)

"""
A model instance can again be created by mapping an input `vector`, which now has 6 entries.
"""
instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

"""
This `instance` contains each of the model components we defined above. 

The argument names input into the `Collection` define the attribute names of the `instance`:
"""
print("Instance Parameters \n")
print("x (Gaussian) = ", instance.gaussian.centre)
print("normalization (Gaussian) = ", instance.gaussian.normalization)
print("sigma (Gaussian) = ", instance.gaussian.sigma)
print("x (Exponential) = ", instance.exponential.centre)
print("normalization (Exponential) = ", instance.exponential.normalization)
print("sigma (Exponential) = ", instance.exponential.rate)

"""
The `Analysis` class above assumed the `instance` contained only a single model-component.

We update its `log_likelihood_function` to use both model components in the `instance` to fit the data.
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

        The data is fitted using an `instance` of multiple 1D profiles (e.g. a `Gaussian`, `Exponential`) where
        their `model_data_1d_via_xvalues_from` methods are called and sumed in order to create a model data
        representation that is fitted to the data.
        """

        """
        The `instance` that comes into this method is an instance of the `Gaussian` and `Exponential` models above, 
        which were created via `af.Collection()`. 
        
        It contains instances of every class we instantiated it with, where each instance is named following the names
        given to the Collection, which in this example is a `Gaussian` (with name `gaussian) and Exponential (with 
        name `exponential`).
        
        The parameter values are again chosen by the non-linear search, based on where it thinks the high likelihood 
        regions of parameter space are. The lines of Python code are commented out below to prevent excessive print 
        statements. 
        """

        # print("Gaussian Instance:")
        # print("Centre = ", instance.gaussian.centre)
        # print("Normalization = ", instance.gaussian.normalization)
        # print("Sigma = ", instance.gaussian.sigma)

        # print("Exponential Instance:")
        # print("Centre = ", instance.exponential.centre)
        # print("Normalization = ", instance.exponential.normalization)
        # print("Rate = ", instance.exponential.rate)

        """
        Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
        """
        xvalues = np.arange(self.data.shape[0])

        """
        Internally, the `instance` variable is a list of all model components pass to the `Collection` above.
        
        we can therefore iterate over them and use their `model_data_1d_via_xvalues_from` methods to create the
        summed overall model data.
        """
        model_data = sum(
            [
                profile_1d.model_data_1d_via_xvalues_from(xvalues=xvalues)
                for profile_1d in instance
            ]
        )

        """
        Fit the model gaussian line data to the observed data, computing the residuals, chi-squared and log likelihood.
        """
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


"""
We can now fit this model to the data using the same API we did before.
"""
analysis = Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(
    nlive=100,
    sample="rwalk",
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
The `info` attribute shows the result in a readable format, showing that all 6 free parameters were fitted for.
"""
print(result.info)

"""
We can again use the max log likelihood instance to visualize the model data of the best fit model compared to the
data.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Cookbooks__

This overview shows the basics of model-fitting with **PyAutoFit**.

The API is designed to be intuitive and extensible, and you should have a good feeling for how you would define
and compose your own model, fit it to data with a chosen non-linear search, and use the results to interpret the
fit.

The following cookbooks give a concise API reference for using **PyAutoFit**, and you should use them as you define
your own model to get a fit going:

- Model Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html
- Searches Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html
- Analysis Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html
- Results Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html

__What Next?__

The next overview describes how to set up a scientific workflow, where many other tasks required to perform detailed but
scalable model-fitting can be delegated to **PyAutoFit**. 

After that, we'll give a run-through of **PyAutoFit**'s advanced statistical inference features, including tools
to scale Bayesian Hierarchical Analysis to large datasets.
"""
