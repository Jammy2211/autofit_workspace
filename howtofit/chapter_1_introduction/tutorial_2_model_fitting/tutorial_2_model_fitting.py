# %%
"""
__Model Fitting__

In this tutorial, we'll fit the Gaussian model from the previous tutorial to the data we loaded.
"""

# %%
#%matplotlib inline

# %%
import autofit as af

import numpy as np

# %%
"""
To begin, lets load the dataset again.

You need to change the path below to the chapter 1 directory so we can load the dataset.
"""

# %%
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# %%
"""
These setup the configs as we did in the previous tutorial.
"""

# %%
af.conf.instance = af.conf.Config(config_path=chapter_path + "config")

dataset_path = chapter_path + "dataset/gaussian_x1/"

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
)

# %%
"""
From here on, we're going to perform all visualization using the 'plot' package, which contains functions for
plotting our line dataset as well as other aspects of the model we'll cover later.

By storing all of our visualization in one package, it will make visualization of our model-fits simpler in
later tutorials.
"""

# %%
from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.plot import (
    dataset_plots,
)

dataset_plots.data(dataset=dataset)
dataset_plots.noise_map(dataset=dataset)

# %%
"""
So, how do we actually go about fitting our Gaussian model to this data? First, we need to be able to generate
an image of our 2D Gaussian model.
"""

# %%
from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.model import gaussian

# %%
"""
Checkout the file:

'autofit_workspace/howtofit/chapter_1_introduction/tutorial_2_model_fitting/src/model/gaussian.py'.

Here, we've extended the Gaussian class to have a method "line_from_values". Given an input set of x coordinates
this computes the intensity of the Gaussian at every point. Our data contains the xvalues we'll use, which are
a 1D NumPy array spanning values 0 to 100.
"""

# %%
print(dataset.xvalues)

# %%
"""
If we pass these values to an instance of the Gaussian class, we can create a line of the gaussian's values.
"""

# %%
model = af.PriorModel(gaussian.Gaussian)

gaussian = model.instance_from_vector(vector=[60.0, 20.0, 15.0])

model_data = gaussian.line_from_xvalues(xvalues=dataset.xvalues)

from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.plot import line_plots

line_plots.line(xvalues=dataset.xvalues, line=model_data, ylabel="Model Data")

# %%
"""
Different values of centre, intensity and sigma change the Gaussian's apperance - have a go at editing some of the
values below.
"""

# %%
gaussian = model.instance_from_vector(vector=[50.0, 10.0, 5.0])
model_data = gaussian.line_from_xvalues(xvalues=dataset.xvalues)
line_plots.line(xvalues=dataset.xvalues, line=model_data, ylabel="Model Data")

# %%
"""
Okay, so lets recap. We've defined a model which is a 1D Gaussian and given a set of parameters for that model
(x, I, sigma) we can create 'model_data' of the Gaussian. And, we have some data of a Gaussian we want to
fit this model with. So how do we do that?

Simple, we take the image from our data and our model_image of the Gaussian and subtract the two to get a
'residual-map'.
"""

# %%
residual_map = dataset.data - model_data
line_plots.line(xvalues=dataset.xvalues, line=residual_map, ylabel="Residual Map")

# %%
"""
Clearly, this model isn't a good fit to the data - which was to be expected as they looked nothing alike!

Next, we want to quantify how good (or bad) the fit actually was, via some goodness-of-fit measure. This measure
needs to account for noise in the data - after all if we fit a pixel badly simply because it was very noisy we want
our goodness-of-fit to account for that.

To account for noise, we take our residual-map and divide it by the noise map, to get the 'normalized residual-map'.
"""

# %%
normalized_residual_map = residual_map / dataset.noise_map
line_plots.line(
    xvalues=dataset.xvalues,
    line=normalized_residual_map,
    ylabel="Normalized Residual Map",
)

# %%
"""
We're getting close to a goodness-of-fit measure, but there is still a problem - we have negative and positive values
in the normalized residual map. A value of -0.2 represents just as good of a fit as a value of 0.2, so we want them
to both be the same value.

Thus, we next define a 'chi-squared map', which is the normalized residual-map squared. This makes negative and
positive values both positive and thus defined on a common overall scale.
"""

# %%
chi_squared_map = (normalized_residual_map) ** 2
line_plots.line(xvalues=dataset.xvalues, line=chi_squared_map, ylabel="Chi-Squared Map")

# %%
"""
Great, even when looking at a chi-squared map its clear that our model gives a rubbish fit to the data.

Finally, we want to reduce all the information in our chi-squared map into a single goodness-of-fit measure. To do
this we define the 'chi-squared', which is the sum of all values on the chi-squared map.
"""

# %%
chi_squared = np.sum(chi_squared_map)
print("Chi-squared = ", chi_squared)

# %%
"""
Thus, the lower our chi-squared, the fewer residuals in the fit between our model and the data and therefore the
better our fit!

From the chi-squared we can then define our final goodness-of-fit measure, the 'log_likelihood', which is the
chi-squared value times -0.5.
"""

# %%
log_likelihood = -0.5 * chi_squared
print("Log Likelihood = ", log_likelihood)

# %%
"""
Why is the log likelihood the chi-squared times -0.5? Lets not worry about. This is simply the standard definition of a
log_likelihood in statistics (it relates to the noise-properties of our data-set). For now, just accept that this is what
a log likelihood is and if we want to fit a model to data our goal is to thus find the combination of model parameters
that maximizes our log_likelihood.

There is a second quantity that enters the log likelihood, called the 'noise-normalization'. This is the log sum of all
noise values squared in our data-set (give the noise map doesn't change the noise_normalization is the same value for
all models that we fit).
"""

# %%
noise_normalization = np.sum(np.log(2 * np.pi * dataset.noise_map ** 2.0))

# %%
"""
Again, like the definition of a log likelihood, lets not worry about why a noise normalization is defined in this way or
why its in our goodness-of-fit. Lets just accept for now that this is how it is in statistics.

Thus, we now have the definition of a log likelihood that we'll use hereafter in all PyAutoFit tutorials.
"""

# %%
log_likelihood = -0.5 * chi_squared + noise_normalization
print("Log Likelihood = ", log_likelihood)

# %%
"""
If you are familiar with model-fitting, you'll have probably heard of terms like 'residuals', 'chi-squared' and
'log_likelihood' before. These are the standard metrics by which a model-fit's quality is measured. They are used for
model fitting in general, so not just when your data is 1D but when its a 2D image, 3D datacube)or something else
entirely!

If you haven't performed model fitting before and these terms are new to you, make sure you are clear on exactly what
they all mean as they are at the core of all model-fitting performed in PyAutoFit!

It was a lot of code performing the fits above and creating our residuals, chi-squareds and likelihoods.

From here on we'll a class to do this, which can be found in the file:
#
'autofit_workspace/howtofit/chapter_1_introduction/tutorial_2_model_fitting/fit/fit.py'

We'll use a 'fit.py' module in all remaining tutorials - for a model-fitting problem its not surprising that we need
a module specific to fitting!
"""

# %%
from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.fit import fit as f

fit = f.DatasetFit(dataset=dataset, model_data=model_data)

print("Fit: \n")
print(fit)
print("Model Data:\n")
print(fit.model_data)
print()
print("Residual Map:\n")
print(fit.residual_map)
print()
print("Chi-Squareds Map:\n")
print(fit.chi_squared_map)
print("Likelihood:")
print(fit.log_likelihood)

# %%
"""
In the plot module, we've created simple tools for plotting different components of a fit. Again, setting up our
plotting in this way will make visualization of our model a lot more straight forward in future tutorials.
"""

# %%
from howtofit.chapter_1_introduction.tutorial_2_model_fitting.src.plot import fit_plots

fit_plots.residual_map(fit=fit)
fit_plots.normalized_residual_map(fit=fit)
fit_plots.chi_squared_map(fit=fit)

# %%
"""
So to recap the previous tutorial and this one:

- We can define a model components in PyAutoFit, like our Gaussian, using Python classes that follow a certain format.
- The model component's parameters each have priors, which given a unit vector can be mapped to an instance of the
  Gaussian class.
- We can use this model instance to create model data of our Gaussian and compare it to data and quantify the
  goodness-of-fit via a log likelihood.

Thus we have everything we need to fit our model to our data! So, how do we go about finding the best-fit model?
That is, the model which maximizes the log likelihood.

The most simple thing we can do is guess parameters, and when we guess parameters that give a good fit, guess another
set of parameters near those values. We can then repeat this process, over and over, until we find a really good model!

For our Gaussian this works pretty well, below I've fitted 5 diferent Gaussian models and ended up landing on
the best-fit model (the model I used to create the dataset in the first place!).
"""

# %%
gaussian = model.instance_from_vector(vector=[50.0, 10.0, 5.0])
model_data = gaussian.line_from_xvalues(xvalues=dataset.xvalues)
fit = f.DatasetFit(dataset=dataset, model_data=model_data)
fit_plots.chi_squared_map(fit=fit)
print("Likelihood:")
print(fit.log_likelihood)

gaussian = model.instance_from_vector(vector=[50.0, 25.0, 5.0])
model_data = gaussian.line_from_xvalues(xvalues=dataset.xvalues)
fit = f.DatasetFit(dataset=dataset, model_data=model_data)
fit_plots.chi_squared_map(fit=fit)
print("Likelihood:")
print(fit.log_likelihood)

gaussian = model.instance_from_vector(vector=[50.0, 25.0, 10.0])
model_data = gaussian.line_from_xvalues(xvalues=dataset.xvalues)
fit = f.DatasetFit(dataset=dataset, model_data=model_data)
fit_plots.chi_squared_map(fit=fit)
print("Likelihood:")
print(fit.log_likelihood)

# %%
"""
You can now perform model-fitting with PyAutoFit! All we have to do is guess lots of parameters, over and over and
over again, until we hit a model with a high log_likelihood. Yay!

Of course, you're probably thinking, is that really it? Should we really be guessing models to find the best-fit?

Obviously, the answer is no. Imagine our model was more complex, that it had many more parameters than just 4.
Our approach of guessing parameters won't work - it could take days, maybe years, to find models with a high
log_likelihood, and how could you even be sure they ware the best-fit models? Maybe a set of parameters you never tried
provide an even better fit?

Of course, there is a much better way to perform model-fitting, and in the next tutorial we'll take you through how
to do such fitting in PyAutoFit, using whats called a 'non-linear search'.


## Datasets and Fits ###

In this tutorial, we relied heavily on PyAutoArray to take care of our data-set. We had a 'dataset.py' module which
contained the data, but it inherited from PyAutoArray and thus handled all visualization for us. We're going to keep
using PyAutoArray to handle our data, but I will make it explcit in templates where I ancitipate where you'll need to
change code to to make it appropriate for data specific to your model-fitting problem. Afterall, there is no
general template we can give you for any data-set.

So, just bare in mind from here on, whenever we use PyAutoArray to load dat or visualize it, it'll be on
you to do that yourself in you code!

To perform fits, we made a new module, 'fit.py'. This module is written in a general way, and it should be nothing
more than a copy and paste job for you to be able to reuse it for your model fitting problem, regardless of the
structure of your data! In tutorial 4, we'll extend 'fit.py' to include data masking.


###Your Model ###

To end, its worth quickly thinking about the model you ultimately want to fit with PyAutoFit. In this example,
we extended the Gaussian class to contain the function we needed to generate an image of the Gaussian and thus
generate the model-image we need to fit our data. For your model fitting problem can you do something similar?
Or is your model-fitting task a bit more complicated than this? Maybe there are more model component you want to
combine or there is an inter-dependency between models?

PyAutoFit provides a lot of flexibility in how you ultimate use your model instances, so whatever your problem you
should find that it is straight forward to find a solution. But, whatever you need to do at its core your modeling
problem will break down into the tasks we did in this tutorial:
#
1) Use your model to create some model data.
2) Subtract it from the data to create residuals.
3) Use these residuals in conjunction with your noise map to define a log likelihood.
4) Find the highest log likelihood models.

So, get thinking about how these steps would be performed for your model!
"""
