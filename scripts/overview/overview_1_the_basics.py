"""
Overview: The Basics
--------------------

**PyAutoFit** is a Python based probabilistic programming language for model fitting and Bayesian inference
of large datasets.

The basic **PyAutoFit** API allows us a user to quickly compose a probabilistic model and fit it to data via a
log likelihood function, using a range of non-linear search algorithms (e.g. MCMC, nested sampling).

This overview gives a run through of:

 - **Models**: Use Python classes to compose the model which is fitted to data.
 - **Instances**: Create instances of the model via its Python class.
 - **Analysis**: Define an ``Analysis`` class which includes the log likelihood function that fits the model to the data.
 - **Searches**: Choose an MCMC, nested sampling or maximum likelihood estimator non-linear search algorithm that fits the model to the data.
 - **Model Fit**: Fit the model to the data using the chosen non-linear search, with on-the-fly results and visualization.
 - **Results**: Use the results of the search to interpret and visualize the model fit.
 - **Samples**: Use the samples of the search to inspect the parameter samples and visualize the probability density function of the results.
 - **Multiple Datasets**: Dedicated support for simultaneously fitting multiple datasets, enabling scalable analysis of large datasets.

This overviews provides a high level of the basic API, with more advanced functionality described in the following
overviews and the **PyAutoFit** cookbooks.

To begin, lets import ``autofit`` (and ``numpy``) using the convention below:
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
from os import path

"""
__Example Use Case__

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a 1D Gaussian profile to noisy data.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
We plot the data with error bars below, showing the noisy 1D signal.
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

 ``x``: The x-axis coordinate where the ``Gaussian`` is evaluated.

 ``N``: The overall normalization of the Gaussian.

 ``sigma``: Describes the size of the Gaussian.

Our modeling task is to fit the data with a 1D Gaussian and recover its parameters (``x``, ``N``, ``sigma``).

__Model__

We therefore need to define a 1D Gaussian as a **PyAutoFit** model.

We do this by writing it as the following Python class:
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

    def model_data_from(self, xvalues: np.ndarray) -> np.ndarray:
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
The **PyAutoFit** model above uses the following format:

- The name of the class is the name of the model, in this case, "Gaussian".

- The input arguments of the constructor (the ``__init__`` method) are the parameters of the model, in this case ``centre``, ``normalization`` and ``sigma``.
  
- The default values of the input arguments define whether a parameter is a single-valued ``float`` or a multi-valued ``tuple``. In this case, all 3 input parameters are floats.
  
- It includes functions associated with that model component, which are used when fitting the model to data.

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
The priors can be manually altered as follows, noting that these updated priors will be used below when we fit the
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
The example above uses the most basic PyAutoFit API to compose a simple model. The API is highly extensible and
can scale to models with thousands of parameters, complex hierarchies and relationships between parameters.
A complete overview is given in the `model cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_.

__Instances__

Instances of a **PyAutoFit** model (created via `af.Model`) can be generated by mapping an input `vector` of parameter 
values to create an instance of the model's Python class.

To define the input `vector` correctly, we need to know the order of parameters in the model. This information is 
contained in the model's `paths` attribute.
"""
print(model.paths)

"""
We input values for the three free parameters of our model in the order specified by the `paths` 
attribute (i.e., `centre=30.0`, `normalization=2.0`, and `sigma=3.0`):
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

"""
This is an instance of the ``Gaussian`` class.
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
We can use functions associated with the class, specifically the `model_data_from` function, to 
create a realization of the `Gaussian` and plot it.
"""
xvalues = np.arange(0.0, 100.0, 1.0)

model_data = instance.model_data_from(xvalues=xvalues)

plt.plot(xvalues, model_data, color="r")
plt.title("1D Gaussian Model Data.")
plt.xlabel("x values of profile")
plt.ylabel("Gaussian Value")
plt.show()
plt.clf()

"""
This "model mapping", whereby models map to an instances of their Python classes, is integral to the core **PyAutoFit**
API for model composition and fitting.

Mapping models to instance of their Python classes is an integral part of the core **PyAutoFit** API. It enables
the advanced model composition and results management tools illustrated in the following overviews and cookbooks.

__Analysis__

We now tell **PyAutoFit** how to fit the model to the data.

We define an `Analysis` class, which includes:

- An `__init__` constructor that takes `data` and `noise_map` as inputs (this can be extended with additional elements 
  necessary for fitting the model to the data).
  
- A `log_likelihood_function` that defines how to fit an `instance` of the model to the data and return a log 
  likelihood value.

Read the comments and docstrings of the `Analysis` class in detail for a full description of how the analysis works.
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

        The data is fitted using an `instance` of the `Gaussian` class where its `model_data_from`
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
        model_data = instance.model_data_from(xvalues=xvalues)

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
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
The `Analysis` class shown above is the simplest example possible. The API is highly extensible and can include
model-specific output, visualization and latent variable calculations. A complete overview is given in the
analysis cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html>`_.

__Non Linear Search__

We now have a model ready to fit the data and an analysis class that performs this fit.

Next, we need to select a fitting algorithm, known as a "non-linear search," to fit the model to the data.

**PyAutoFit** supports various non-linear searches, which can be broadly categorized into three types: 
MCMC (Markov Chain Monte Carlo), nested sampling, and maximum likelihood estimators.

For this example, we will use the nested sampling algorithm called Dynesty.
"""
search = af.DynestyStatic(
    nlive=100,  # Example how to customize the search settings
)

"""
The default settings of the non-linear search are specified in the configuration files of **PyAutoFit**, just
like the default priors of the model components above. The ensures the basic API of your code is concise and
readable, but with the flexibility to customize the search to your specific model-fitting problem.

PyAutoFit supports a wide range of non-linear searches, including detailed visualuzation, support for parallel
processing, and GPU and gradient based methods using the library JAX (https://jax.readthedocs.io/en/latest/).
A complete overview is given in the `searches cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html>`_.

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
model_data = result.max_log_likelihood_instance.model_data_from(
    xvalues=np.arange(data.shape[0])
)

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
plt.plot(xvalues, model_data, color="r")
plt.title("Dynesty model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Samples__

The results object also contains a ``Samples`` object, which contains all information on the non-linear search.

This includes parameter samples, log likelihood values, posterior information and results internal to the specific
algorithm (e.g. the internal dynesty samples).

Below we use the samples to plot the probability density function cornerplot of the results.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
The `results cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html>`_ also provides 
a run through of the samples object API.

__Multiple Datasets__

Many model-fitting problems require multiple datasets to be fitted simultaneously in order to provide the best
constraints on the model.

In **PyAutoFit**, all you have to do to fit multiple datasets is sum your ``Analysis`` classes together:
"""
# For illustration purposes, we'll input the same data and noise-map as the example, but for a realistic example
# you would input different datasets and noise-maps to each analysis.

analysis_0 = Analysis(data=data, noise_map=noise_map)
analysis_1 = Analysis(data=data, noise_map=noise_map)

# This means the model is fitted to both datasets simultaneously.

analysis = analysis_0 + analysis_1

# summing a list of analysis objects is also a valid API:

analysis = sum([analysis_0, analysis_1])

"""
__Wrap Up__

This overview covers the basic functionality of **PyAutoFit** using a simple model, dataset, and model-fitting problem, 
demonstrating the fundamental aspects of its API.

By now, you should have a clear understanding of how to define and compose your own models, fit them to data using 
a non-linear search, and interpret the results.

The **PyAutoFit** API introduced here is highly extensible and customizable, making it adaptable to a wide range 
of model-fitting problems.

The next overview will delve into setting up a scientific workflow with **PyAutoFit**, utilizing its API to 
optimize model-fitting efficiency and scalability for large datasets. This approach ensures that detailed scientific 
interpretation of the results remains feasible and insightful.

__Resources__

The `autofit_workspace: <https://github.com/Jammy2211/autofit_workspace/>`_ repository on GitHub provides numerous 
examples demonstrating more complex model-fitting tasks.

This includes cookbooks, which provide a concise reference guide to the **PyAutoFit** API for advanced model-fitting:

- [Model Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html): Learn how to compose complex models using multiple Python classes, lists, dictionaries, NumPy arrays and customize their parameterization. 

- [Analysis Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html): Customize the analysis with model-specific output and visualization to gain deeper insights into your model fits. 

- [Searches Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html): Choose from a variety of non-linear searches and customize their behavior. This includes options like outputting results to hard disk and parallelizing the search process. 

- [Results Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html): Explore the various results available from a fit, such as parameter estimates, error estimates, model comparison metrics, and customizable visualizations. 

- [Configs Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/configs.html): Customize default settings using configuration files. This allows you to set priors, search settings, visualization preferences, and more. 

- [Multiple Dataset Cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/multiple_datasets.html): Learn how to fit multiple datasets simultaneously by combining their analysis classes so that their log likelihoods are summed. 

These cookbooks provide detailed guides and examples to help you leverage the **PyAutoFit** API effectively for a wide range of model-fitting tasks.

__Extending Models__

The main overview is now complete, howeveer below we provide an example of how to compose and fit a model
consisting of multiple components, which is a common requirement in many model-fitting problems.

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
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
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

    def model_data_from(self, xvalues: np.ndarray):
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
        their `model_data_from` methods are called and sumed in order to create a model data
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
        
        we can therefore iterate over them and use their `model_data_from` methods to create the
        summed overall model data.
        """
        model_data = sum(
            [profile_1d.model_data_from(xvalues=xvalues) for profile_1d in instance]
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

model_gaussian = instance.gaussian.model_data_from(xvalues=np.arange(data.shape[0]))
model_exponential = instance.exponential.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

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
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
Finish.
"""
