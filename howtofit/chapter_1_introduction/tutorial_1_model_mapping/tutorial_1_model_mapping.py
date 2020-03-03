import autofit as af
import autoarray as aa
import autoarray.plot as aplt

# In this tutorial, we'll parameterize a simple model and use PyAutoFit to map its parameters to a model instance,
# which we'll ultimately need to fit it to data.

# To get a feeling for our model lets look at the data we'll ultimately be fitting, a 2D Gaussian.

# You need to change the path below to the chapter 1 directory so we can load the dataset.
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Lets setup the config files for this tutorial. As we covered in the previous tutorial, these configure how
# visualization appears, but they also customize our model as we'll describe in this tutorial.
af.conf.instance = af.conf.Config(config_path=chapter_path + "/config")

# The dataset path specifies where the dataset is located, this time in the directory 'chapter_path/dataset'. We'll
# load the example dataset containing one Gaussian.
dataset_path = chapter_path + "dataset/gaussian_x1/"

# We now load this dataset from .fits files and create an instance of an 'imaging' object.
imaging = aa.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    pixel_scales=0.1,
)

aplt.imaging.image(imaging=imaging)

# Its not until tutorial 3 that we'll actually this image with a model. But its worth us looking at it now so we can
# understand the model we're going to fit. So what is the model?

# Clearly, its a two dimensional Gaussian defined as:

# g(r) = (I / (sigma * sqrt(2*pi)) * exp (-0.5 * (r / sigma)^2)

# Where:

# I - Describes the intensity of the Gaussian.
# sigma - Describes the size of the Gaussian.
# r - The radial coordinate where the Gaussian is evaluated r = sqrt(y^2 + x^2).

# This simple equation describes our model - a 2D Gaussian - which has 4 parameters, (y, x, I, sigma). Using different
# values of these 4 parameters we can describe *any* possible 2D Gaussian.

# At its core, PyAutoFit is all about making it simple to define a model, like a 2D Gaussian, and straight forwardly
# map a set of input parameters to the model.

# So lets go ahead and create our model of a 2D Gaussian. Take a look at the file

# 'autofit_workspace/howtofit/chapter_1_introduction.tutorial_1_model_mapping/model/gaussians.py'.

# Here we define our 2D Gaussian model using the code:


class Gaussian:
    def __init__(
        self,
        centre=(0.0, 0.0),  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma


# The class's format is key, as this format is how PyAutoFit requires the components of a model to be written, where:

# - The name of the class is the name of the model component, in this case, "Gaussian".
# - The input arguments of the constructor are the model parameters which we will ultimately fit for.
# - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float (e.g. like
#   the intensity and sigma) or a multi-valued tuple (e.g. like the centre).

# By writing a model component in this way, we can use the Python class to set it up as model in PyAutoFit.

from howtofit.chapter_1_introduction.tutorial_1_model_mapping.model import gaussians

model = af.PriorModel(gaussians.Gaussian)

# The model is what PyAutoFit calls a PriorModel - we'll explain the name below.
print("PriorModel Gaussian object: \n")
print(model)

# Using our model we can create an 'instance' of the model, by mapping a list of physical values of each parameter to
# the model.
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0])

# The instance is an instance of the Gaussian class.
print("Model Instance: \n")
print(instance)

# It has the parameters of our Gaussian with the values input above.
print("Instance Parameters \n")
print("y = ", instance.centre[0])
print("x = ", instance.centre[1])
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# Note that above, the centre's default input value was a 2D tuple (0.0, 0.0). The mapper creates the model instance
# with the centre as a tuple.

# It has the parameters of our Gaussian with the values input above.
print("Instance Centre \n")
print("y = ", instance.centre)

# Congratulations! You've defined your first model in PyAutoFit! :)

# So, why is it called a PriorModel?

# The parameters of a PriorModel in PyAutoFit all have a prior associated with them. Priors encode our expectations on
# what values we expect each parameter can have. For example, we might know that our Gaussian will be centred near
# (0.0, 0.0).

# Where are priors set? Checkout the config file:

# 'autofit_workspace/howtofit/chapter_1_introduction/config/priors/default/gaussians.ini

# For our Gaussian, we use the following default priors:

# centre_0 (y) - GaussianPrior centred on (0.0) and with width (0.3).
# centre_1 (x) - GaussianPrior centred on (0.0) and with width (0.3).
# intensity (I) - LogUniformPrior (base 10) between 0.000001 and 100000.0
# sigma - UniformPrior between 0.0 and 10.0

# Config files in PyAutoFit use the module name to read the config files. This is why our Gaussian component is in the
# the module "gaussians.py", so that PyAutoFit knows to look for config files with the name "gaussians.ini".

# So, when are these priors actually used? They are used to generate model instances from a unit-vector, a vector
# defined in the same way as the physical vector above but with values spanning from 0 -> 1.

# Unit values are mapped to physical values using the prior, for example:

# For a UniformPrior defined between 0.0 and 10.0:

# - An input unit value of 0.5 will give the physical value 5.0.
# - An input unit value of 0.8 will give te physical value 8.0.

# For a LogUniformPrior (base 10) defined between 1.0 and 100.0:

# - An input unit value of 0.5 will give the physical value 10.0.
# - An input unit value of 1.0 will give te physical value 100.0.

# For a GauassianPrior defined with mean 1.0 and sigma 1.0:

# - An input unit value of 0.5 (e.g. the centre of the Gaussian) will give the physical value 1.0.
# - An input unit value of 0.8173 (e.g. 1 sigma confidence) will give te physical value 1.9051.

# Lets take a look:
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.3, 0.8])

# The instance is again an instance of the Gaussian class.
print("Model Instance: \n")
print(instance)

# It has physical values for the parameters mapped from the priors defined in the gaussians.ini config file.
print("Instance Parameters \n")
print("y = ", instance.centre[0])
print("x = ", instance.centre[1])
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# We can overwrite the priors defined in the config.
model.centre.centre_0 = af.UniformPrior(lower_limit=10.0, upper_limit=20.0)
model.centre.centre_1 = af.UniformPrior(lower_limit=50.0, upper_limit=60.0)
model.intensity = af.GaussianPrior(mean=5.0, sigma=7.0)
model.sigma = af.LogUniformPrior(lower_limit=1.0, upper_limit=100.0)

# Our model, with all new priors, can again be used to map unit values to create a model instance.
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.3, 0.8])

# Its physical values are mapped using the new priors defined above and not those in the gaussians.ini config file.
print("Instance Parameters \n")
print("y = ", instance.centre[0])
print("x = ", instance.centre[1])
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# We can also set physical limits on parameters, such that a model instance cannot generate parameters outside of a
# specified range.

# For example, a Gaussian cannot has a negative intensity, so we can set its lower limit to a value of 0.0.
model = af.PriorModel(gaussians.Gaussian)
model.intensity = af.GaussianPrior(
    mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1000.0
)

# The unit vector input below creates a negative intensity value, so the line below leads PyAutoFit to raise an error.
# instance = model.instance_from_unit_vector(unit_vector=[0.01, 0.01, 0.01, 0.01])

# The config file 'autofit_workspace/howtofit/chapter_1_introduction/config/priors/limits/gaussians.ini' set the
# default limits on all parameters of our model.


# And with that, you've completed tutorial 1!

# At this point, you might be wondering, whats the big deal? Sure, its cool that we set up a model and its nice that
# we can translate priors to parameters in this way, but how is this actually going to help me perform model fitting?
# With a bit of effort couldn't I have written some code to do this myself?

# Well, you're probably right, but this tutorial is covering just the backend of PyAutoFit, what holds everything
# together if you will. Once you start using PyAutoFit, you're ultimately never going to directly perform model
# mapping yourself, its the 'magic' behind the scenes that makes model-fitting work.

# So, we're pretty much ready to move on to tutorial 2, where we'll actually fit this model to some data. However,
# first, I want you to quickly think about the model you want to fit. How would you write it as a class using the
# PyAutoFit format above? what are the free parameters of you model? Are there multiple model components you are going
# to want to fit to your data?

# Below are two more classes one might use to perform model fitting, the first is the model of a linear-regression line
# of the form y = mx + c that you might fit to a 1D data-set:


class LinearFit:
    def __init__(self, gradient=1.0, intercept=0.0):

        self.gradient = gradient
        self.intercept = intercept


# The second example is one from Astronomy. When fitting the light of a galaxy, a common analytic function used by
# Astronomers is the "EllipticalSersic" function, which ahs 7 free parameters:


class EllipticalSersic:
    def __init__(
        self,
        centre=(0.0, 0.0),
        axis_ratio=1.0,
        phi=0.0,
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=4.0,
    ):

        self.centre = centre
        self.axis_ratio = axis_ratio
        self.phi = phi
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index
