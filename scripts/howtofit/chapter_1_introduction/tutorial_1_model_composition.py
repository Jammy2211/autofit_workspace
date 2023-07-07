"""
Tutorial 1: Models
==================

At the heart of model-fitting is the model: a set of equations, numerical processes and assumptions describing a
physical system of interest. The goal of model-fitting is to understand this physical system more, ultimately
develop more complex models which describe more aspects of the system more accurately.

In Astronomy, a model may describe the distribution of stars within a galaxy. In biology, is may describe the
interaction of proteins within a cell. In finance, it may describe the evolution of stock prices in a market.
Your model depends on your topic of study, but in all cases the model acts as the mathematical description of
some physical system you seek understand better, and hope ultimately to make new predictions of.

Whatever your model, the equations that underpin will be defined by "free parameters". Changing these parameters
changes the prediction of the model.

For example, an Astronomy model of the distribution of stars may contain a
parameter describing the brightness of the stars, a second parameter defining their number density and a third
parameter describing their colors. If we multiplied the parameter describribing the brightness of the stars by 2,
the stars would therefore appear twice as bright.

Once the model (e.g. the undrlying equations) is defined and a values for the free parameters have been chosen, the
model can create "model data". This data is a realization of how the physical system appears for that model with
those parameters.

For example, a model of the distribution of stars within a galaxy can be used to create a model image of that galaxy.
By changing the parameters governing the distribution of stars, it can produce many different model images, with
different brightness, colors, sizes, etc.

In this tutorial, we will learn the basics of defining a model, and we will in particular:

 - Define a simple model, described by few single equations.

 - Show that this model is described by 3 or more free parameters.

 - Use the model, with different sets of parameters, to generate model data.

This will all be performed using the **PyAutoFit** API for model composition, which forms the basis of all model
fitting performed by **PyAutoFit**.
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

If you don't have an `autofit_workspace` (perhaps you cloned / forked the **PyAutoFit** GitHub repository?) you can
download it here:
 
 https://github.com/Jammy2211/autofit_workspace

At the top of every tutorial notebook, you will see the following cell. 

This cell uses the project `pyprojroot` to locate the path to the workspace on your computer and use it to set the 
working directory of the notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

"""
__Model Parameterization__

A model is a set of equations, numerical processes and assumptions that describe a physical system and dataset.

We can pretty much consider anything is a model. In this example, our model will simply be one or more 1 dimensional
Gaussian, defined by the following equaiton:

\begin{equation*}
g(x, I, \sigma) = \frac{N}{\sigma\sqrt{2\pi}} \exp{(-0.5 (x / \sigma)^2)}
\end{equation*}

Where:

`x`: Is the x-axis coordinate where the `Gaussian` is evaluated.

`N`: Describes the overall normalization of the Gaussian.

$\sigma$: Describes the size of the Gaussian (Full Width Half Maximum = $\mathrm {FWHM}$ = $2{\sqrt {2\ln 2}}\;\sigma$)

Whilst a 1D Gaussian may seem like a somewhat rudimentary model, it actually has a lot of real-world applicaiton
in signal process, where 1D Gausians are fitted to 1D datasets in order to quantify the size of a signal. Our
model is therefore a realstic representation of a real world modeling problrem!

We therefore now have a model, which as expected is a set of equations (just one in this case) that describes a 
dataset.

The model has 3 parameters, $(x, N, \sigma)$, where using different combinations of these parameters creates different 
realizations of the model.

So, how do we compose this model is **PyAutoFit**?

__Model Composition__

To define a "model component" in **PyAutoFit** we simply write it as a Python class using the format shown below:
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
The format of this Python class defines how **PyAutoFit** will compose the `Gaussian` as a model, where:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor (the `__init__` method) are the parameters of the model, in the
  example above `centre`, `normalization` and `sigma`.
  
- The default values of the input arguments define whether a parameter is a single-valued `float` or a 
  multi-valued `tuple`. For the `Gaussian` class above, no input parameters are a tuple, but later examples use tuples. 
  
- It includes functions associated with that model component, specifically the `model_data_1d_via_xvalues_from` 
  function. When we create instances of a `Gaussian` below, this is used to generate 1D representation of it as a 
  NumPy array.

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

This shows that ech model parameter has an associated prior, which are described fully in tutorial 3 of this chapter.

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
We input values for the 3 free parameters of our model following the order of paths 
above (`centre=30.0`, `normalization=2.0` and `sigma=3.0`), creating an `instance` of the `Gaussian` via the model.
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
__More Complex Models__

The code above seemed like a lot of work just to create an instance of the `Guassian` class. Couldn't we have
just done the following instead?

 `instance = Gaussian(centre=30.0, normalization=2.0, sigma=3.0)`.
 
Yes, we could have. 

However, the model composition API used above is designed to make composing complex models, consisting of multiple 
components with many free parameters, straightforward and scalable.

To illustrate this, lets end the tutorial by composing a model made of multiple Gaussians and also another 1D
profile, an Exponential, which is defined following the equation:

\begin{equation*}
g(x, I, \lambda) = N \lambda \exp{- \lambda x }
\end{equation*}

Where:

`x`: Is the x-axis coordinate where the `Exponential` is evaluated.

`N`: Describes the overall normalization of the `Exponential`

$\lambda$: Describes the rate of decay of the exponential.

We first define the `Exponential` using the same format as above. 
"""


class Exponential:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Exponential`s model parameters.
        rate=0.01,
    ):
        """
        Represents a 1D Exponential profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to fit example datasets
        via a non-linear search.

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
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )


"""
We can easily compose a model consisting of 1 `Gaussian` object and 1 `Exponential` object using the `af.Collection`
object:
"""
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

"""
All of the information about the model created via the collection can be printed at once using its `info` attribute:
"""
print(model.info)

"""
Because the `Gaussian` and `Exponential` are being passed to a `Collection` they are automatically 
assigned as `Model` objects.

We can therefore omit the `af.Model` method when passing classes to a `Collection`, making the Python code more
concise and readable.
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
The `model.info` appears identical to the previous example.
"""
print(model.info)

"""
A `Collection` behaves analogous to a `Model`, but it contains a multiple model components.

We can see this by printing its `paths` attribute, where paths to all 6 free parameters via both model components
are shown.

The paths have the entries `.gaussian.` and `.exponential.`, which correspond to the names we input into  
the `af.Collection` above. 

If the input `gaussian=` were changed to `gaussian_edited=`, this will be reflected in the `paths` below.
"""
print(model.paths)

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
In the context of the equations that define the model, the model is simply the sum of the two equations that define
the `Gaussian` and `Exponential`.

Generating the `model_data` therefore requires us to simply sum each individual model component`s `model_data`, which
we do and visualize below.
"""
xvalues = np.arange(0.0, 100.0, 1.0)

model_data_0 = instance.gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)
model_data_1 = instance.exponential.model_data_1d_via_xvalues_from(xvalues=xvalues)

model_data = model_data_0 + model_data_1

plt.plot(xvalues, model_data, color="r")
plt.plot(xvalues, model_data_0, "b", "--")
plt.plot(xvalues, model_data_1, "k", "--")
plt.title("1D Gaussian + Exponential Model Data.")
plt.xlabel("x values of profile")
plt.ylabel("Value")
plt.show()
plt.clf()

"""
__Extensibility__

It is hopefully now clear why we use `Model` and `Collection` objects to compose our model.

They can easily be extended to compose complex models with many components and parameters. For example, we could
input more `Gaussian` and `Exponential` components into the `Collection`, or we could write new Python classes
that represent new model components with more parameters.

These objects serve many other key purposes that we will cover in later tutorials, 

__Wrap Up__

In this tutorial, we introduced how to define and compose a model, which we can generate model data from. 

To end, have a think about your particular field of study and the problem you are hoping to solve through 
model-fitting., What is the model you might want to fit? What Python class using the format above are requird to
compose the right model? What are the free parameters of you model?

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


"""
Finish.
"""
