"""
Overview: The Basics
--------------------

PyAutoFit is a Python based probabilistic programming language which enables detailed Bayesian analysis of scientific
datasets, with everything necessary to scale-up Bayesian analysis to very complex models and extremely large datasets.

In this overview, we introduce the basic API for model-fitting with **PyAutoFit**, including how to define a likelihood
function, compose a probabilistic model, and fit it to data via a non-linear fitting algorithm (e.g.
Markov Chain Monta Carlo (MCMC), maximum likelihood estimator, nested sampling).

If you have previous experience performing model-fitting and Bayesian inference there will be nothing new here, but
we'll highlight some benefits of using **PyAutoFit** instead of setting up the modeling manually
yourself (e.g. by wrapping an MCMC library with your likelihood function).

The biggest benefits of using **PyAutoFit** are presented in the next two overviews once the API is clear. The first
focuses on the "scientific workflow", which streamlines the Bayesian analysis of large datasets. The second gives a
run through of numerous statistical inference methods, for example Bayesian hierarchical analysis.
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
__Example: Noisy 1D Signal__

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

We therefore need to define a 1D Gaussian as a model component in **PyAutoFit**.

A model component is written as a Python class in *PyAutoFit** using the following format:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor (the `__init__` method) are the parameters of the model, in the
  example above `centre`, `normalization` and `sigma`.
  
- The default values of the input arguments define whether a parameter is a single-valued `float` or a 
  multi-valued `tuple`. For the `Gaussian` class above, no input parameters are a tuple, but later examples use tuples. 
  
- It includes functions associated with that model component, specifically the `model_data_1d_via_xvalues_from` 
  function. When we create instances of a `Gaussian` below, this is used to generate 1D representation of it as a 
  NumPy array.
"""
class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

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

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the grid.
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

This shows that ech model parameter has an associated prior.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
__Model Mapping__

Instances of model components created via the `af.Model` object can be created, where an input `vector` of
parameters is mapped to the Python class the model object was created using.

We first need to know the order of parameters in the model, so we know how to define the input `vector`. This
information is contained in the models `paths` attribute:
"""
print(model.paths)

"""
We input values for the 3 free parameters of our model following the order of paths above (`centre=30.0`, 
`normalization=2.0` and `sigma=3.0`), creating an `instance` of the model Gaussian. 

The notion of a model mapping to an `instance` in **PyAutoFit** is an important one, and is used below to fit the model
to data.
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
__Analysis__

Now we've defined a model which can be mapped to instances of PYthon classes, we need to tell **PyAutoFit** how to 
fit the model to data. 

We therefore define an `Analysis` class, which includes:

 - An `__init__` constructor which takes as input the `data`, `noise_map` and can be extended to include any other 
 quantities necessary to fit the model to the data.

 - A `log_likelihood_function` defining how given an `instance` of the model we fit it to the data and return a 
 log likelihood value.

Read the comments and docstrings of the `Analysis` object below in detail for more insights into how this object
works.
"""