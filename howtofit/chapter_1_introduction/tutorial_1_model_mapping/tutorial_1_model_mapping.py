# %%
"""
__Model Mapping__

In this tutorial, we'll parameterize a simple model and use PyAutoFit to map its parameters to a model instance,
which we'll ultimately need to fit to data.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autofit as af
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
To get a feeling for our model lets look at the data we'll ultimately be fitting, a 1D Gaussian.

You need to change the path below to the workspace directory so we can load the dataset.
"""

# %%
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace"
chapter_path = f"{workspace_path}/howtofit/chapter_1_introduction"

# %%
"""
You're going to see a line like the one below (with 'conf.instance =') in every tutorial this chapter. This sets the
following property:

    - The path to the configuration files used by PyAutoFit. You need to give the path to your autofit_workspace, so 
      the configuration files in the workspace are used (e.g. '/path/to/autolens_workspace/config'). 

(These will work autommatically if the WORKSPACE environment variable was set up correctly during installation. 
Nevertheless, setting the paths explicitly within the code is good practise.
"""

# %%
conf.instance = conf.Config(config_path=f"{chapter_path}/config")

# %%
"""
The dataset path specifies where the dataset is located, this time in the directory 'chapter_path/dataset'. We'll
load the example dataset containing one Gaussian.
"""

# %%
dataset_path = f"{chapter_path}/dataset/gaussian_x1/"

# %%
"""
We now load the data and noise from .fits files, where:

- The data is a 1D numpy array of values corresponding to the observed counts of the Gaussian.
- The noise map corresponds to the expected noise in every data point.
"""

# %%
data_path = f"{dataset_path}/data.fits"
data_hdu_list = fits.open(data_path)
data = np.array(data_hdu_list[0].data)

noise_map_path = f"{dataset_path}/noise_map.fits"
noise_map_hdu_list = fits.open(noise_map_path)
noise_map = np.array(noise_map_hdu_list[0].data)

# %%
"""
Lets plot the Gaussian using Matplotlib. 

The Gaussian is on a line of xvalues, which we'll compute using the shape of the Gaussian data and plot on the x-axis.
These xvalues will be used in later tutorials to create and fit Gaussians to the data.
"""

# %%
xvalues = np.arange(data.shape[0])
plt.plot(xvalues, data)
plt.show()

# %%
"""
Its not until tutorial 3 that we'll actually fit this image with a model. But its worth us looking at it now so we
can understand the model we're going to fit. So what is the model?

Clearly, its a one-dimensional Gaussian defined as:

g(x, I, sigma) = (I / (sigma * sqrt(2*pi)) * exp (-0.5 * (x / sigma)^2)

Where:

x - Is x-axis coordinate where the Gaussian is evaluated.
I - Describes the intensity of the Gaussian.
sigma - Describes the size of the Gaussian.

This simple equation describes our model - a 1D Gaussian - and it has 3 parameters, (x, I, sigma). Using different
values of these 3 parameters we can describe *any* possible 1D Gaussian.

At its core, PyAutoFit is all about making it simple to define a model and straight forwardly map a set of input
parameters to the model.

So lets go ahead and create our model of a 1D Gaussian.
"""

# %%
class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma

    def line_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the light profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : ndarray
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


# %%
"""
The class's format is how PyAutoFit requires the components of a model to be written, where:

- The name of the class is the name of the model component, in this case, "Gaussian".
- The input arguments of the constructor are the model parameters which we will ultimately fit for, in this case the
  centre, intensity and sigma.
- The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued floats or a 
  multi-valued tuple. For the Gaussian class, no input parameters are a tuple and we will show an example of a tuple 
  input in a later tutorial).
  
By writing a model component in this way, we can use the Python class to set it up as model component in PyAutoFit.
PyAutoFit can the generate model components as instances of their Python class, meaning that its functions 
(e.g. 'line_from_xvalues') are accessible to PyAutoFit.

To set it up as a model component, we use a PriorModel object.
"""

# %%
model = af.PriorModel(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=np.inf)
model.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=np.inf)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=np.inf)
print("PriorModel Gaussian object: \n")
print(model)

# %%
"""
Using this PriorModel we can create an 'instance' of the model, by mapping a list of physical values of each parameter 
as follows.
"""

# %%
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0])

# %%
"""
The instance is an instance of the Gaussian class.
"""

# %%
print("Model Instance: \n")
print(instance)

# %%
"""
It has the parameters of the Gaussian with the values input above.
"""

# %%
print("Instance Parameters \n")
print("x = ", instance.centre)
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# %%
"""
Congratulations! You've defined your first model in PyAutoFit! :)

So, why is it called a PriorModel?

The parameters of a PriorModel in PyAutoFit all have a prior associated with them. Priors encode our expectations on
what values we expect each parameter can have. For example, we might know that our Gaussian will be centred near 0.0.

How are priors set? In this example, we did not specify priors, so they default to UniformPriors between 0 and 1. Below,
we'll show how to customize priors, and in a later tutorial we'll explain how for a given model component we can 
define config files that specify the default priors.

### How Are Priors Used? ###

Priors are used to create model instances from a unit-vector, which is a vector defined in the same way as the vector 
above but with values spanning from 0 -> 1.

Unit values are mapped to physical values via the prior, for example:

For a UniformPrior defined between 0.0 and 10.0:

- An input unit value of 0.5 will give the physical value 5.0.
- An input unit value of 0.8 will give te physical value 8.0.

For a LogUniformPrior (base 10) defined between 1.0 and 100.0:

- An input unit value of 0.5 will give the physical value 10.0.
- An input unit value of 1.0 will give te physical value 100.0.

For a GauassianPrior defined with mean 1.0 and sigma 1.0:

- An input unit value of 0.5 (e.g. the centre of the Gaussian) will give the physical value 1.0.
- An input unit value of 0.8173 (e.g. 1 sigma confidence) will give te physical value 1.9051.

Lets take a look:
"""

# %%
"""
We can overwrite the default priors assumed for each parameter.
"""

# %%
model.centre = af.UniformPrior(lower_limit=10.0, upper_limit=20.0)
model.intensity = af.GaussianPrior(mean=5.0, sigma=7.0)
model.sigma = af.LogUniformPrior(lower_limit=1.0, upper_limit=100.0)

# %%
"""
These priors are now used to map our unit values to physical values when we create an instance of the Gaussian
class.
"""

# %%
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.3, 0.8])

# %%
"""
Lets check that this instance is again an instance of the Gaussian class.
"""

# %%
print("Model Instance: \n")
print(instance)

# %%
"""
It now has physical values for the parameters mapped from the priors defined above.
"""

# %%
print("Instance Parameters \n")
print("x = ", instance.centre)
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# %%
"""
We can also set physical limits on parameters, such that a model instance cannot generate parameters outside of a
specified range.

For example, a Gaussian cannot have a negative intensity, so we can set its lower limit to a value of 0.0.
"""

# %%
model.intensity = af.GaussianPrior(
    mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1000.0
)

# %%
"""
The unit vector input below creates a negative intensity value, such that if you uncomment the line below PyAutoFit 
raises an error.
"""

# %%
# instance = model.instance_from_unit_vector(unit_vector=[0.01, 0.01, 0.01])

# %%
"""
In a later tutorial, we'll explain how config files can again be used to set the default limits of every parameter.


And with that, you've completed tutorial 1!

At this point, you might be wondering, whats the big deal? Sure, its cool that we set up a model and its nice that
we can translate priors to parameters in this way, but how is this actually going to help me perform model fitting?
With a bit of effort couldn't I have written some code to do this myself?

Well, you're probably right, but this tutorial is covering just the backend of PyAutoFit - what holds everything
together. Once you start using PyAutoFit, its unlikely that you'll perform model mapping yourself, its the 'magic' 
behind the scenes that makes model-fitting work.

So, we're pretty much ready to move on to tutorial 2, where we'll actually fit this model to some data. However,
first, I want you to quickly think about the model you want to fit. How would you write it as a class using the
PyAutoFit format above? What are the free parameters of you model? Are there multiple model components you are going
to want to fit to your data?

Below are two more classes one might use to perform model fitting, the first is the model of a linear-regression line
of the form y = mx + c that you might fit to a 1D data-set:
"""

# %%
class LinearFit:
    def __init__(self, gradient=1.0, intercept=0.0):

        self.gradient = gradient
        self.intercept = intercept


# %%
"""
The second example is a two-dimensional Gaussian. Here, the centre now has two coordinates (y,x), which in PyAutoFit
is more suitably defined using a tuple.
"""

# %%
class Gaussian2D:
    def __init__(self, centre=(0.0, 0.0), intensity=0.1, sigma=1.0):

        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma
