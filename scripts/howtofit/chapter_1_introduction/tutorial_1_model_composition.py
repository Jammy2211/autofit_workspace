"""
Tutorial 1: Model Composition
=============================

In this tutorial, we'll compose a simple model and show how **PyAutoFit** allows us to generate instances of the model,
which will allow us to fit the model to data in later tutorials.
"""
import autofit as af
import numpy as np
import matplotlib.pyplot as plt

"""
__Paths__

**PyAutoFit** assumes the current working directory is `/path/to/autofit_workspace/` on your hard-disk (or in Binder). 
This is so that it can:
 
 - Load configuration settings from config files in the `autofit_workspace/config` folder.
 - Load example data from the `autofit_workspace/dataset` folder.
 - Output the results of models fits to your hard-disk to the `autofit/output` folder. 

If you don't have an autofit_workspace (perhaps you cloned / forked the **PyAutoFit** GitHub repository?) you can
download it here:
 
 https://github.com/Jammy2211/autofit_workspace

At the top of every tutorial notebook, you'll see the following cell. This cell uses the project `pyprojroot` to
locate the path to the workspace on your computer and use it to set the working directory of the notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

"""
__Data__

Throughout these tutorials we will fit noisy 1D data containing a signal, where the signal was generated using a 
Gaussian. 

These are loaded from .json files, where:

 - The data is a 1D numpy array of values corresponding to the observed counts of the Gaussian.
 - The noise-map corresponds to the expected noise value in every data point.
 
These datasets were created using the scripts in `autofit_workspace/howtofit/simulators`, feel free to check them out!
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the `Gaussian` via Matplotlib. 

The 1D signal is observed on a line of uniformly spaced xvalues, which we'll compute using the shape of the data and 
plot as the x-axis. 
"""
xvalues = np.arange(data.shape[0])
plt.plot(xvalues, data, color="k")
plt.title("1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Signal Normalization")
plt.show()

"""
We can also plot its `noise_map` (which in this example are all constant values) as a standalone 1D plot or
as error bars on the `data`.
"""
plt.plot(xvalues, noise_map, color="k")
plt.title("Noise-map")
plt.xlabel("x values of noise-map")
plt.ylabel("Noise-map value (Root mean square error)")
plt.show()

plt.errorbar(
    xvalues, data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian dataset with errors from the noise-map.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()

"""
__Model Parameterization__

We now wish to define a model that can fit the signal in this data. 

What model could fit this data? The obvious choice is a one-dimensional `Gaussian` defined as:

\begin{equation*}
g(x, I, \sigma) = \frac{N}{\sigma\sqrt{2\pi}} \exp{(-0.5 (x / \sigma)^2)}
\end{equation*}

Where:

x - Is an x-axis coordinate where the `Gaussian` is evaluated.
N - Describes the overall normalization of the Gaussian.
$\sigma$ - Describes the size of the Gaussian.

This simple equation describes our model, a 1D `Gaussian`, and it has 3 parameters, $(x, N, \sigma)$. Using different
values of these 3 parameters we can create a realization of any 1D Gaussian.

__Model Composition__

We now compose the 1D `Gaussian` above as a model in **PyAutoFit**. 

First, we write a Python class as follows:
"""


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian`s model parameters.
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray):
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
The format of this Python class defines how **PyAutoFit** will compose it as a model, where:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor are the parameters of the model, which we will vary when we fit it to our
 data. In this case, the free parameters are `centre`, `normalization` and `sigma`.
  
- The default values of the input arguments tell **PyAutoFit** whether a parameter is a single-valued `float` or a 
  multi-valued `tuple`. For the `Gaussian` class, no input parameters are a tuple and we will show an example of a 
  tuple  input in a later tutorial). 
  
- It includes functions associated with that model component, for example the `model_data_1d_via_xvalues_from` function, which
  allows an instance of a `Gaussian` to create its 1D representation as a NumPy array.

To compose the model using the `Gaussian` class above we use the **PyAutoFit** `Model` object.
"""
model = af.Model(Gaussian)
print("Model `Gaussian` object: \n")
print(model)

"""
We can inspect the model to note that it indeed has a total of 3 parameters (why this is called `prior_count` will be
explained in tutorial 3):
"""
print(model.prior_count)

"""
All of the information about the model can be printed at once using its `info` attribute:
"""
print(model.info)

"""
__Model Mapping__

At the core of **PyAutoFit** is how it map Python classes that are set up via the `Model` object to instances of
that Python classes, where the values of its parameters are set during this mapping. 

For example, we below create an `instance` of the model, by mapping a list of physical values of each parameter as 
follows.
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
We can use the functions associated with this class, specifically the `model_data_1d_via_xvalues_from` function, to create a 
realization of the Gaussian and plot it.
"""
realization = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

plt.plot(xvalues, realization, color="r")
plt.title("1D Gaussian Realization.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
__Discussion__

Whilst the simple example above is informative, it may be somewhat underwhelming. Afterall, how difficult would it 
have been to write Python code that defines this `Gaussian` with those values yourself? Why bother setting up the
`Gaussian` as a `Model` and using the `model.instance_from_vector` command when you could of just set up an instance
of the Gaussian by passing it the parameters manually yourself, e.g.:

`instance = Gaussian(centre=30.0, normalization=2.0, sigma=3.0)`

The reason, is because model composition and mapping get very complicated very quickly. In tutorial 3, we'll introduce 
priors, which also need to be fully for when performing model mapping. Later tutorials will compose models with 10+ 
parameters, which require high levels of customization in their parameterization. At the end of chapter 1 we'll use 
**PyAutoFit** to build multi-level models from hierarchies of Python classes that potentially comprise hundreds of 
parameters!

Therefore, things might seem somewhat and unnecessary right now, but the tools we're covering now will enable very 
complex models to be composed, mapped and fitted by the end of this chapter! 

__Wrap Up__

In this tutorial, we introduced how to parameterize and compose a model and define priors for each of its parameters.
This used Python classes and allowed us to map input values to instances of this Python class. Nothing we've introduced 
in this tutorial was particularly remarkable, but it has presented us with the core interface we'll use to do advanced
model fitting in later tutorials. 

Finally, quickly think about a model you might want to fit. How would you write it as a Python class using the format 
above? What are the free parameters of you model? Are there multiple model components you are going to want to fit to 
your data data?

If you decide to add a new model-component to the `autofit_workspace` specific to your model-fitting task, first
checkout the following script, which explains how to set up the **PyAutoFit** configuration files associated with 
your model.

`autofit_workspace/*/overview/new_model_component/new_model_component.ipynb`

Below are two more example Python classes one might define to perform model fitting, the first is the model of a 
linear-regression line of the form $y = mx + c$ that you might fit to a 1D data-set:
"""


class LinearFit:
    def __init__(self, gradient=1.0, intercept=0.0):
        self.gradient = gradient
        self.intercept = intercept


"""
The second example is a two-dimensional Gaussian. Here, the centre now has two coordinates (y,x), which in 
**PyAutoFit** is more suitably defined using a tuple.
"""


class Gaussian2D:
    def __init__(self, centre=(0.0, 0.0), normalization=0.1, sigma=1.0):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma
