import autofit as af

import numpy as np

# In this tutorial, we'll fit the Gaussian model from the previous tutorial to the data we loaded.

# To begin, lets load the dataset again.

# You need to change the path below to the chapter 1 directory so we can load the dataset.
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# These setup the configs as we did in the previous tutorial.
af.conf.instance = af.conf.Config(config_path=chapter_path + "/config")

dataset_path = chapter_path + "dataset/gaussian_x1/"

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

# From here on, we're going to perform all visualization using the 'plot' package, which contains functions for
# plotting our line dataset as well as other aspects of the model we'll cover later.

# By storing all of our visualization in one package, it will make visualization of our model-fits simpler in
# later tutorials.

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.plot import (
    dataset_plots,
)

dataset_plots.data(dataset=dataset)
dataset_plots.noise_map(dataset=dataset)

# So, how do we actually go about fitting our Gaussian model to this data? First, we need to be able to generate
# an image of our 2D Gaussian model.

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.model import gaussian

# Checkout the file:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_2_model_fitting/src/model/gaussian.py'.

# Here, we've extended the Gaussian class to have a method "line_from_values". Given an input set of x coordinates
# this computes the intensity of the Gaussian at every point. Our data contains the xvalues we'll use, which are
# a 1D NumPy array spanning values 0 to 100.
print(dataset.xvalues)

# We can use PyAutoArray to create such a grid, which we'll make the same dimensions as our data above.

# The "pixel-scales" define the conversion between pixels (which range from values of 0 to 100) and Gaussian
# coordinates (which define the length dimensions of its centre and sigma).
grid = aa.grid.uniform(shape_2d=dataset.shape_2d, pixel_scales=dataset.pixel_scales)

# This grid is a uniform 2D array of coordinates in length units of our Gaussian profile.

# We can print each coordinate of this grid, revealing that it consists of coordinates where the spacing between each
# coordinate corresponds to the 'pixel_scale' of 1.0 we defined above
print("(y,x) pixel 0:")
print(grid.in_2d[0, 0])
print("(y,x) pixel 1:")
print(grid.in_2d[0, 1])
print("(y,x) pixel 2:")
print(grid.in_2d[0, 2])
print("(y,x) pixel 100:")
print(grid.in_2d[1, 0])
print("etc.")

# Grids in PyAutoArray are stored as both 1D and 2D NumPy arrays, because different calculations benefit from us
# using the array in different formats. We can access both the 1D and 2D arrays automatically by specifying the input
# as a 1D or 2D NumPy index.
print("(y,x) pixel 0 (accessed in 2D):")
print(grid.in_2d[0, 0])
print("(y,x) pixel 0 (accessed in 1D):")
print(grid.in_1d[0])

# The shape of the grid is also available in 1D and 2D, consisting of 625 (25 x 25) coordinates.
print(grid.shape_2d)
print(grid.shape_1d)

# We can print the entire grid in either 1D or 2D.
print(grid.in_2d)
print(grid.in_1d)

# If we pass this grid to an instance of the Gaussian class, we can create an image of the gaussian.

model = af.PriorModel(gaussians.Gaussian)

gaussian = model.instance_from_vector(vector=[0.0, 0.0, 1.0, 1.0])

model_image = gaussian.line_from_xvalues(grid=grid)

# Much like the grid, the arrays PyAutoArray computes are accessible in both 2D and 1D.
print(model_image.shape_2d)
print(model_image.shape_1d)
print(model_image.in_2d[0, 0])
print(model_image.in_1d[0])
print(model_image.in_2d)
print(model_image.in_1d)

# PyAutoArray has the tools we need to visualize the Gaussian's image.
aplt.array(array=model_image)

# Different values of centre, intensity and sigma change the Gaussian's apperance - have a go at editing some of the
# values below.
gaussian = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0])
model_image = gaussian.line_from_xvalues(grid=grid)
aplt.array(array=model_image)

# Okay, so lets recap. We've defined a model which is a 2D Gaussian and given a set of parameters for that model
# (y, x, I, sigma) we can create a 'model_image' of the Gaussian. And, we have some data of a Gaussian we want to
# fit this model with. So how do we do that?

# Simple, we take the image from our data and our model_image of the Gaussian and subtract the two to get a
# 'residual-map'.

residual_map = dataset.image - model_image
aplt.array(array=residual_map)

# Clearly, this model isn't a good fit to the data - which was to be expected as they looked nothing alike!

# Next, we want to quantify how good (or bad) the fit actually was, via some goodness-of-fit measure. This measure
# needs to account for noise in the data - after all if we fit a pixel badly simply because it was very noisy we want
# our goodness-of-fit to account for that.

# To account for noise, we take our residual-map and divide it by the noise-map, to get the 'normalized residual-map'.
normalized_residual_map = residual_map / dataset.noise_map
aplt.array(array=normalized_residual_map)

# We're getting close to a goodness-of-fit measure, but there is still a problem - we have negative and positive values
# in the normalized residual map. A value of -0.2 represents just as good of a fit as a value of 0.2, so we want them
# to both be the same value.

# Thus, we next define a 'chi-squared map', which is the normalized residual-map squared. This makes negative and
# positive values both positive and thus defined on a common overall scale.
chi_squared_map = (normalized_residual_map) ** 2
aplt.array(array=chi_squared_map)

# Great, even when looking at a chi-squared map its clear that our model gives a rubbish fit to the data.

# Finally, we want to reduce all the information in our chi-squared map into a single goodness-of-fit measure. To do
# this we define the 'chi-squared', which is the sum of all values on the chi-squared map.
chi_squared = np.sum(chi_squared_map)
print("Chi-squared = ", chi_squared)

# Thus, the lower our chi-squared, the fewer residuals in the fit between our model and the data and therefore the
# better our fit!

# From the chi-squared we can then define our final goodness-of-fit measure, the 'likelihood', which is the
# chi-squared value times -0.5.
likelihood = -0.5 * chi_squared
print("Likelihood = ", likelihood)

# Why is the likelihood the chi-squared times -0.5? Lets not worry about. This is simply the standard definition of a
# likelihood in statistics (it relates to the noise-properties of our data-set). For now, just accept that this is what
# a likelihood is and if we want to fit a model to data our goal is to thus find the combination of model parameters
# that maximizes our likelihood.

# There is a second quantity that enters the likelihood, called the 'noise-normalization'. This is the log sum of all
# noise values squared in our data-set (give the noise-map doesn't change the noise_normalization is the same value for
# all models that we fit).
noise_normalization = np.sum(np.log(2 * np.pi * dataset.noise_map ** 2.0))

# Again, like the definition of a likelihood, lets not worry about why a noise normalization is defined in this way or
# why its in our goodness-of-fit. Lets just accept for now that this is how it is in statistics.

# Thus, we now have the definition of a likelihood that we'll use hereafter in all PyAutoFit tutorials.
likelihood = -0.5 * chi_squared + noise_normalization
print("Likelihood = ", likelihood)

# If you are familiar with model-fitting, you'll have probably heard of terms like 'residuals', 'chi-squared' and
# 'likelihood' before. These are the standard metrics by which a model-fit's quality is measured. They are used for
# model fitting in general, so not just when your data is an image but when its 1D data (e.g a line), 3D data
# (e.g. a datacube) or something else entirely!

# If you haven't performed model fitting before and these terms are new to you, make sure you are clear on exactly what
# they all mean as they are at the core of all model-fitting performed in PyAutoFit!

# It was a lot of code performing the fits above and creating our residuals, chi-squareds and likelihoods.

# From here on we'll a class to do this, which can be found in the file:
#
# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_2_model_fitting/fit/fit.py'

# We'll use a 'fit.py' module in all remaining tutorials - for a model-fitting problem its not surprising that we need
# a module specific to fitting!

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.fit import fit as f

fit = f.DatasetFit(dataset=dataset, model_data=model_image)

print("Fit: \n")
print(fit)
print("Model-Image:\n")
print(fit.model_data.in_2d)
print(fit.model_data.in_1d)
print()
print("Residual Maps:\n")
print(fit.residual_map.in_2d)
print(fit.residual_map.in_1d)
print()
print("Chi-Squareds Maps:\n")
print(fit.chi_squared_map.in_2d)
print(fit.chi_squared_map.in_1d)
print("Likelihood:")
print(fit.likelihood)

# PyAutoArray provides the tools we need to visualize a fit.
aplt.fit_imaging.subplot_fit_imaging(fit=fit)

# So to recap the previous tutorial and this one:

# - We can define a model components in PyAutoFit, like our Gaussian, using Python classes that follow a certain format.
# - The model component's parameters each have priors, which given a unit vecto can be mapped to an instance of the
#   Gaussian class.
# - We can use this model-instance to create a model-image of our Gaussian and compare it to data and quantify the
#   goodness-of-fit via a likelihood.

# Thus we have everything we need to fit our model to our data! So, how do we go about finding the best-fit model?
# That is, the model which maximizes the likelihood.

# The most simple thing we can do is guess parameters, and when we guess parameters that give a good fit, guess another
# set of parameters near those values. We can then repeat this process, over and over, until we find a really good model!

# For our Gaussian this works pretty well, below I've fitted 5 diferent Gaussian models and ended up landing on
# the best-fit model (the model I used to create the dataset in the first place!).
gaussian = model.instance_from_vector(vector=[0.0, 0.5, 3.0, 3.0])
model_image = gaussian.line_from_xvalues(grid=grid)
fit = f.DatasetFit(dataset=dataset, model_data=model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=fit)
print("Likelihood:")
print(fit.likelihood)

gaussian = model.instance_from_vector(vector=[0.0, 0.0, 3.0, 3.0])
model_image = gaussian.line_from_xvalues(grid=grid)
fit = f.DatasetFit(dataset=dataset, model_data=model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=fit)
print("Likelihood:")
print(fit.likelihood)

gaussian = model.instance_from_vector(vector=[0.0, 0.0, 10.0, 3.0])
model_image = gaussian.line_from_xvalues(grid=grid)
fit = f.DatasetFit(dataset=dataset, model_data=model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=fit)
print("Likelihood:")
print(fit.likelihood)

gaussian = model.instance_from_vector(vector=[0.0, 0.0, 10.0, 1.0])
model_image = gaussian.line_from_xvalues(grid=grid)
fit = f.DatasetFit(dataset=dataset, model_data=model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=fit)
print("Likelihood:")
print(fit.likelihood)

gaussian = model.instance_from_vector(vector=[0.0, 0.0, 10.0, 5.0])
model_image = gaussian.line_from_xvalues(grid=grid)
fit = f.DatasetFit(dataset=dataset, model_data=model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=fit)
print("Likelihood:")
print(fit.likelihood)

# You can now perform model-fitting with PyAutoFit! All we have to do is guess lots of parameters, over and over and
# over again, until we hit a model with a high likelihood. Yay!

# Of course, you're probably thinking, is that really it? Should we really be guessing models to find the best-fit?

# Obviously, the answer is no. Imagine our model was more complex, that it had many more parameters than just 4.
# Our approach of guessing parameters won't work - it could take days, maybe years, to find models with a high
# likelihood, and how could you even be sure they ware the best-fit models? Maybe a set of parameters you never tried
# provide an even better fit?

# Of course, there is a much better way to perform model-fitting, and in the next tutorial we'll take you through how
# to do such fitting in PyAutoFit, using whats called a 'non-linear search'.


### Datasets and Fits ###

# In this tutorial, we relied heavily on PyAutoArray to take care of our data-set. We had a 'dataset.py' module which
# contained the data, but it inherited from PyAutoArray and thus handled all visualization for us. We're going to keep
# using PyAutoArray to handle our data, but I will make it explcit in templates where I ancitipate where you'll need to
# change code to to make it appropriate for data specific to your model-fitting problem. Afterall, there is no
# general template we can give you for any data-set.

# So, just bare in mind from here on, whenever we use PyAutoArray to load dat or visualize it, it'll be on
# you to do that yourself in you code!

# To perform fits, we made a new module, 'fit.py'. This module is written in a general way, and it should be nothing
# more than a copy and paste job for you to be able to reuse it for your model fitting problem, regardless of the
# structure of your data! In tutorial 4, we'll extend 'fit.py' to include data masking.


#### Your Model ###

# To end, its worth quickly thinking about the model you ultimately want to fit with PyAutoFit. In this example,
# we extended the Gaussian class to contain the function we needed to generate an image of the Gaussian and thus
# generate the model-image we need to fit our data. For your model fitting problem can you do something similar?
# Or is your model-fitting task a bit more complicated than this? Maybe there are more model component you want to
# combine or there is an inter-dependency between models?

# PyAutoFit provides a lot of flexibility in how you ultimate use your model instances, so whatever your problem you
# should find that it is straight forward to find a solution. But, whatever you need to do at its core your modeling
# problem will break down into the tasks we did in this tutorial:
#
# 1) Use your model to create some model data.
# 2) Subtract it from the data to create residuals.
# 3) Use these residuals in conjunction with your noise-map to define a likelihood.
# 4) Find the highest likelihood models.

# So, get thinking about how these steps would be performed for your model!
