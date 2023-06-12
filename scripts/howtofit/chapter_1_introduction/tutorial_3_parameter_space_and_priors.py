"""
Tutorial 3: Parameter Space And Priors
======================================

In the previous tutorial, we define fitting functions that allowed us to create model-data using realizations of a
1D `Gaussian` model and fit it to the data. We achieved a good fit, but only by guessing values of parameters. In the
next two tutorials we are going to learn how to fit a model to data properly, which means we first need to define
concepts of a parameter space and priors.

__Parameter Space__

If mathematics, you will have learnt that we can write a simple function as follows:

$f(x) = x^2$

In this function, when we input the parameter $x`$ in to the function $f$, it returns a value $f(x)$. The mappings
between values of $x$ and $f(x)$ define what we can call the parameter space of this function (and if you remember
your math classes, the parameter space of the function $f(x) = x^2$ is a parabola).

A function can of course have multiple parameters:

$f(x, y, z) = x + y^2 - z^3$

This function has 3 parameters, $x$, $y$ and $z$. The mappings between $x$, $y$ and $z$ and $f(x, y, z)$ define another
parameter space, albeit this parameter space now has 3 dimensions. Nevertheless, just like we could plot a parabola to
visualize the parameter space $f(x) = x^2$, we could visualize this parameter space as 3 dimensional surface.

In the previous tutorial, we used realizations of the `Gaussian` class to fit data with a model so as to return a log
likelihood.

This process can be thought of as us computing a likelihood from a function, just like our functions $f(x)$ above.
However, the log likelihood function is not something that we can write down analytically as an equation and its
behaviour is inherently non-linear. Nevertheless, it is a function, and if we put the same values of model
parameters into this function the same value of log likelihood will be returned.

Therefore, we can write this log likelihood function as follows, where the parameters $(x, N, \sigma)$ are again
the parameters of our `Gaussian`:

$f(x, N, \sigma) = log_likelihood$

By expressing the likelihood in this way we have defined a parameter space! The solutions to this function cannot be
written analytically and it is highly complex and non-linear. However, we have already learnt how we can use this
function to determine a log likelihood, by creating realizations of the Gaussian and comparing them to the data.

__Priors__

We are now thinking about our model and log likelihood function as a parameter space, which is crucial for
understanding how we will fit the model to data in the next tutorial. Before we do that, we need to consider one more
concept, how do we define where in parameter space we search for solutions? What values of model parameters do we
consider viable solutions?

A parameter, say, the `centre` of the `Gaussian`, could in principle take any value between negative and positive
infinity. However, when we inspect the data it is clearly confined to values between 0.0 and 100.0, therefore we should
define a parameter space that only contains these solutions as these are the only physically plausible values
of `centre` (e.g. between 0.0 --> 100.0).

These are called the 'priors'. Our priors define where parameter space has valid solutions, and throughout these
tutorials we will use three types of prior:

- UniformPrior: The permitted values of a parameter are between a `lower_limit` and `upper_limit` and we assign equal
probability to all solutions between these limits. For example, the `centre` of the `Gaussian` will typically assume
a uniform prior between 0.0 and 100.0.

- LogUniformPrior: Like a `UniformPrior` this confines the values of a parameter between a `limit_limit`
and `upper_limit`, but we assign the probability of solutions with a log distribution with base 10. For example,
the `normalization` of the `Gaussian` will typically assume a log uniform prior between 1e-2 and 100.0.

- GaussianPrior: The permitted values of a parameter whose probability is tied to a Gaussian distribution with
a `mean` and width `sigma`. For example, the `sigma` of the `Gaussian` will typically assume Gaussian prior with mean
10.0 and sigma 5.0.
"""
import autofit as af
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

Again, load and plot the dataset from the `autofit_workspace/dataset` folder.
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
        centre: float = 30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization: float = 1.0,  # <- are the Gaussian`s model parameters.3
        sigma: float = 5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray) -> np.ndarray:
        """
        Calculate the normalization of the light profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

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
__Model Mapping via Priors__

We can again use **PyAutoFit** to set the `Gaussian` as a model and map it to instances of the `Gaussian`, however
we can now do this via priors.
"""
model = af.Model(Gaussian)
print("Model `Gaussian` object: \n")
print(model)

"""
We now set the prior for each parameter.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=0.1, upper_limit=10.0)
model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)

"""
The updated priors are reflected in the model's `info` attribute.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
As a quick reminder, we have seen that using this `Model` we can create an `instance` of the model, by mapping a 
list of physical values of each parameter as follows.
"""
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0])
print("Instance Parameters \n")
print("x = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
Priors are used to create model instances via a mapping analogous to the one above, but from a unit-vector of values.

This vector is defined in the same way as the vector above but with values spanning from 0 -> 1, where the unit values 
are mapped to physical values via the prior, for example:

For the `UniformPrior` defined between 0.0 and 100.0:

- An input unit value of 0.5 will give the physical value 5.0.
- An input unit value of 0.8 will give te physical value 80.0.

For the `LogUniformPrior` (base 10) defined between 0.1 and 10.0:

- An input unit value of 0.5 will give the physical value 1.0.
- An input unit value of 1.0 will give te physical value 10.0.

For a `GaussianPrior `defined with mean 10.0 and sigma 5.0:

- An input unit value of 0.5 (e.g. the centre of the Gaussian) will give the physical value 10.0.
- An input unit value of 0.8173 (e.g. 1 sigma confidence) will give te physical value 14.5256.

Lets take a look:
"""
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.3, 0.8173])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("x = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
__How Are Priors Actually Used?__

Priors allow us to map unit vectors to physical parameters and therefore define a parameter space. However, the actual
process of mapping unit-values to physical values in this way is pretty much all handled by **PyAutoFit** 
"behind ths scenes" and is not something you'll explicitly do yourself. Nevertheless, this is core concept of any
model-fitting exercise and is why we have covered it in this tutorial. 

In the next tutorial, we'll see how this mapping between unit and physical values is built-in to the algorithms we 
use to perform model-fitting!

__Limits__

We can also set physical limits on parameters, such that a model instance cannot generate parameters outside of a
specified range.

For example, a `Gaussian` cannot have a negative normalization, so we can set its lower limit to a value of 0.0.

This is what the `gaussian_limits` section in the priors config files sets.
"""
model.normalization = af.GaussianPrior(
    mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1000.0
)

"""
The unit vector input below creates a negative normalization value, such that if you uncomment the line 
below **PyAutoFit** raises an error.
"""
# instance = model.instance_from_unit_vector(unit_vector=[0.01, 0.01, 0.01])

"""
__Prior Configs__

For highly complex models with many parameters, it is cumbersome to have to manually define the prior on every 
parameter and would likely lead to input mistakes.

**PyAutoFit** allows one to define all of the default prior values in a configuration file, such that they are loaded
automatically. This means we do not need manually define the priors ourselves.

To do this, we define the `Gaussian` class in a standalone Python 
module `autofit_workspace/*/howtofit/chapter_1_introduction/gaussian.py` (as opposed to this Python script). 
The name of this module is used to look for a file `gaussian.json` in the directory `autofit_workspace/config/priors` 
such that the default priors of the model are loaded from the file `gaussian.json`. 

For example, because our `Gaussian` is in the module `gaussian.py`, its priors are loaded from the priors config
file `gaussian.json`. Check this file out now to see the default priors; we'll discuss what the different inputs
mean later on.

This is illustrated below, where we are using the `Gaussian` defined in `gaussian.py` and inspect its prior to see
they have been automatically set up via the config file.
"""
import gaussian as g

model = af.Model(g.Gaussian)
print("Model `Gaussian` object via priors configs: \n")
print(model)

"""
__Wrap Up__

In this tutorial, we introduce the notion of a parameter space and priors, which **PyAutoFit**'s model mapping 
utilities map between. We are now in a position to perform a model-fit, which will be the subject of the next tutorial.
 
The description of priors in this tutorial was somewhat of a simplification; we viewed them as a means to map a 
unit values of parameters to physical values. In Bayesian inference, priors play a far more important role, as they
define one's previous knowledge of the model before performing the fit. They directly impact the solution that one
infers and ultimately dictate how the model-fitting is performed.

The aim of the **HowToFit** tutorials is not to teach the reader the details of Bayesian inference but instead set
you up with the tools necessary to perform a model-fit. Nevertheless, it is worth reading up on Bayesian inference and 
priors at any of the following links:

https://seeing-theory.brown.edu/bayesian-inference/index.html

https://towardsdatascience.com/probability-concepts-explained-bayesian-inference-for-parameter-estimation-90e8930e5348
"""
