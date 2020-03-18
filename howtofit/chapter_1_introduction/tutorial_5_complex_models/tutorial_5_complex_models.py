import autofit as af
import numpy as np

from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.model import profiles
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.dataset import (
    dataset as ds,
)

# Up to now, we've fitted a very simple model - a single 1D Gaussian with just 3 free parameters. In this tutorial,
# we'll look at how PyAutoFit allows us to define, build and fit models of arbritrary complexity.

# To begin, you should check out the module 'tutorial_5_complex_models/model/profiles.py'. In previous tutorials this
# module was called 'gaussian.py' and it contained only the Gaussian class we fitted to data. The module now includes
# a second profile, 'Exponential', which like the Gaussian class can be a model-component that is fitted to data.

# Up to now, our data has always been generated using a single Gaussian line profile. Thus, we have only needed to fit
# it with a single Gaussian module. In this tutorial, our datasets are now superpositions of multiple profiles (e.g
# 2+ Gaussians, 2+ Exponentials or a combination of Gaussians and Exponentials). The models we define and fit must
# theerefore also be composed of multiple profiles, such that when we generate the model-data we generate it as a
# superposition of all line profiles in our model.

# You need to change the path below to the chapter 1 directory so we can load the dataset
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

# Lets quickly recap tutorial 1, where using PriorModels we created a Gaussian as a model component and used it to map a
# list of parameters to a model instance.

model = af.PriorModel(profiles.Gaussian)

print("PriorModel Gaussian object: \n")
print(model)

instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("x = ", instance.centre)
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# Defining a model using multiple model components is straight forward in PyAutoFit, using a CollectionPriorModel
# object.

model = af.CollectionPriorModel(
    gaussian=profiles.Gaussian, exponential=profiles.Exponential
)

# A CollectionPriorModel behaves like a PriorModel but for a collection of model components. For example, it can
# create a model instance by mapping a list of parameters, which in this case is 6 (3 for the Gaussian [centre,
# intensity, sigma] and 3 for the Exponential [centre, intensity, rate]).

instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

# This instance contains each of the model components we defined above, using the input argument name of the
# CollectionoPriorModel to define the attributes in the instance:

print("Instance Parameters \n")
print("x (Gaussian) = ", instance.gaussian.centre)
print("intensity (Gaussian) = ", instance.gaussian.intensity)
print("sigma (Gaussian) = ", instance.gaussian.sigma)
print("x (Exponential) = ", instance.exponential.centre)
print("intensity (Exponential) = ", instance.exponential.intensity)
print("sigma (Exponential) = ", instance.exponential.rate)

# We can call the components of a CollectionPriorModel whatever we want, and the mapped instance will use those names.

model = af.CollectionPriorModel(
    a_name=profiles.Gaussian, another_name=profiles.Exponential
)

instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

print("Instance Parameters \n")
print("x (Gaussian) = ", instance.a_name.centre)
print("intensity (Gaussian) = ", instance.a_name.intensity)
print("sigma (Gaussian) = ", instance.a_name.sigma)
print("x (Exponential) = ", instance.another_name.centre)
print("intensity (Exponential) = ", instance.another_name.intensity)
print("sigma (Exponential) = ", instance.another_name.rate)

# Now we can create a model composed of multiple components, lets have a go at fitting it to a dataset. To do this,
# we've had to update the Phase used in this tutorial, specifically changing its 'analysis.py' module to create its
# model data as a super position of all of the model's individual line profiles. For example, in the model above, the
# model data is the sum of the Gaussian's individual line profile and Exponential's individual line profile.

# Checkout 'phase.py' and 'analysis.py' now, for a description of how this has been implemented.

# Lets create the phase and run it to fit a dataset which was specifically generated as a sum of a Gaussian and
# Exponential profile.

dataset_path = chapter_path + "dataset/gaussian_x1_exponential_x1/"

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

# We again need to create a mask for our data. In this exmample, we'll omit actual masking of the dataset, but still
# need to define a mask to pass the 'phase.run' method.

mask = np.full(fill_value=False, shape=dataset.data.shape)

# Lets now perform the fit using our model which is composed of two profiles. You'll note that the MultiNest
# dimensionality has increased from N=3 to N=6, given that we are now fitting two profiles each with 3 free parameters.

from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.phase import (
    phase as ph,
)

phase = ph.Phase(
    phase_name="phase_t5_gaussian_x1_exponential_x1",
    profiles=af.CollectionPriorModel(
        gaussian=profiles.Gaussian, exponential=profiles.Exponential
    ),
)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x1_exponential_x1"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# Quickly inspect the results of this fit, which you may have noticed takes a bit longer to run than the fits performed
# in previous tutorials. This is because the dimensionality of the model we are fitted increased from 3 to 6.


# With the CollectionPriorModel, PyAutoFit gives us all the tools we need to compose and fit any model imaginable!
# For example, lets fit a model composed of two Gaussians nd and an Exponential, which will have a dimensionality of
# N=9.

dataset_path = chapter_path + "dataset/gaussian_x2_exponential_x1/"

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

phase = ph.Phase(
    phase_name="phase_t5_gaussian_x2_exponential_x1",
    profiles=af.CollectionPriorModel(
        gaussian_0=profiles.Gaussian,
        gaussian_1=profiles.Gaussian,
        exponential=profiles.Exponential,
    ),
)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x2_exponential_x1"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# We can fully custommize the model that we fit. Lets suppose we have a dataset that consists of three Gaussian line
# profiles, but we also know the following information about the dataset:

# - All 3 Gaussians are centrally aligned.
# - The sigma of one Gaussian is equal to 1.0.

# We can edit our CollectionPriorModel to meet these constraints accordingly:

model = af.CollectionPriorModel(
    gaussian_0=profiles.Gaussian,
    gaussian_1=profiles.Gaussian,
    gaussian_2=profiles.Gaussian,
)

# This aligns the centres of the 3 Gaussians, reducing the dimensionality of the model from N=9 to N=7
model.gaussian_0.centre = model.gaussian_1.centre
model.gaussian_1.centre = model.gaussian_2.centre

# This fixes the sigma value of one Gaussian to 1.0, further reducing the dimensionality from N=7 to N=6.
model.gaussian_0.sigma = 1.0

# We can now fit this model using a phase as per usual.

dataset_path = chapter_path + "dataset/gaussian_x3/"

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

phase = ph.Phase(phase_name="phase_t5_gaussian_x3", profiles=model)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x3"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# And with that, we are complete. In this tutorial, we learned how to compose complex models in PyAutoFit and adjust our
# 'phase.py' and 'analyis.py' modules to fit them. To end, you should think again in more detail about your model
# fitting problem:

# Are there many different model components you may wish to define and fit?
# Is your model-data the super position of many different model components, like the line profiles in this tutorial?

# In this tutorial, all components of our model did the same thing - represent a 'line' of data. In your model, you may
# have model components that represent different parts of your model, which need to be combined in more complicated ways
# in order to create your model-fit. In such circumstances, the 'fit' method in your 'Analysis' class may be significantly
# more complex than the example shown in this tutorial. Nevertheless, you now have all the tools you need to define,
# compose and fit very complex models - there isn't much left for you to learn on your journey through PyAutoFit!
