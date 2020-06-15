from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np

from autoconf import conf
import autofit as af
from autofit_workspace.examples.complex import model as m
from autofit_workspace.examples.complex import analysis as a

"""
__Example: Result__

In this example, we'll repeat the fit performed in 'fit.py' of 1D data of a Gaussian + Exponential profile with 1D line 
data using the  non-linear  search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files 'example/model.py','example/analysis.py' and 'example/fit.py' to 
see how the fit is performed by the code below. The first section of code below is simply repeating the commands in
'example/fit.py', so feel free to skip over it until you his the __Result__ section.

The attributes of the Result object are described in 'examples/simple/result.py'. This example will not cover the 
attributes in full, and instead only focus on how the use of a more complex model changes the Result object.
"""

# %%
#%matplotlib inline

# %%
"""
__Paths__

Setup the path to the autofit_workspace, using a relative directory name.
"""

# %%
workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

# %%
"""
Use this path to explicitly set the config path and output path.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""
__Data__

First, lets load our data of a 1D Gaussian + Exponential, which we'll plot before we perform the model-fit.
"""

# %%
dataset_path = f"{workspace_path}/dataset/gaussian_x1__exponential_x1"

data_hdu_list = fits.open(f"{dataset_path}/data.fits")
data = np.array(data_hdu_list[0].data)

noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
noise_map = np.array(noise_map_hdu_list[0].data)

# %%
"""
__Model__

Next, we create our model, which in this case corresponds to a Gaussian + Exponential. In model.py, you will have
noted the Gaussian has 3 parameters (centre, intensity and sigma) and Exponential 3 parameters (centre, intensity and
rate). These are the free parameters of our model that the non-linear search fits for, meaning the non-linear
parameter space has dimensionality = 6.

In the simple example tutorial, we used a PriorModel to create the model of the Gaussian. PriorModels cannot be used to
compose models from multiple model components and for this example we must instead use the CollectionPriorModel.
"""

# %%
model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

# %%
"""
Checkout 'autofit_workspace/config/json_priors' - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the non-linear search as shown below.
"""

# %%
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

# %%
"""
__Analysis__

We now set up our Analysis, using the class described in 'analysis.py'. The analysis describes how given an instance
of our model (a Gaussian + Exponential) we fit the data and return a log likelihood value. For this simple example,
we only have to pass it the data and its noise-map.
"""

# %%
analysis = a.Analysis(data=data, noise_map=noise_map)

# %%
"""
Create the non-linear object for emcee and perform the fit.
"""

# %%
emcee = af.Emcee(
    nwalkers=50,
    nsteps=2000,
    initialize_method="ball",
    initialize_ball_lower_limit=0.49,
    initialize_ball_upper_limit=0.51,
    auto_correlation_check_for_convergence=True,
    auto_correlation_check_size=100,
    auto_correlation_required_length=50,
    auto_correlation_change_threshold=0.01,
    number_of_cores=1,
)

result = emcee.fit(model=model, analysis=analysis)

# %%
"""
__RESULT__

Here, we'll look in detail at how the information contained in the result changes when we fit a more complex model. If
you are unfamiliar with the result object, first read through 'examples/simple/result.py'.

First, we can note that the parameters list of lists now has 6 entries in the parameters column, given the 
dimensionality of the model has increased from N=3 to N=6.
"""

# %%
samples = result.samples
print("All Parameters:")
print(samples.parameters)
print("Sample 10's sixth parameter value (Exponential -> rate)")
print(samples.parameters[9][5], "\n")

# %%
"""
The vectors containing models have the same meaning as before, but they are also now of size 6 given the increase in
model complexity.
"""

# %%
print("Result and Error Vectors:")
print(samples.median_pdf_vector)
print(samples.max_log_likelihood_vector)
print(samples.max_log_posterior_vector)
print(samples.vector_at_upper_sigma(sigma=3.0))
print(samples.vector_at_lower_sigma(sigma=3.0))
print(samples.error_vector_at_upper_sigma(sigma=3.0))
print(samples.error_vector_at_lower_sigma(sigma=3.0), "\n")

# %%
"""
The parameter names and labels now contain 6 entries, including the Exponential class that was not included in the
simple model example.
"""

# %%
print(samples.parameter_names)
print(samples.parameter_labels)
print("\n")

# %%
"""
When we return a result as an instance, it provides us with instances of the model using the Python classes used to
compose it. Because our fit uses a CollectionPriorModel (as opposed to a PriorModel in the simple example) the instance
returned a dictionary named acoording to the names given to the CollectionPriorModel, which above were 'gaussian' and
'exponential'.
"""

# %%
max_log_likelihood_instance = samples.max_log_likelihood_instance

print("Max Log Likelihood Gaussian Instance:")
print("Centre = ", max_log_likelihood_instance.gaussian.centre)
print("Intensity = ", max_log_likelihood_instance.gaussian.intensity)
print("Sigma = ", max_log_likelihood_instance.gaussian.sigma, "\n")
print("Max Log Likelihood Exponential Instance:")
print("Centre = ", max_log_likelihood_instance.exponential.centre)
print("Intensity = ", max_log_likelihood_instance.exponential.intensity)
print("Sigma = ", max_log_likelihood_instance.exponential.rate, "\n")

# %%
"""
For our example problem of fitting a 1D Gaussian + Exponential profile, this makes it straight forward to plot 
the maximum likelihood model:
"""

# %%
model_gaussian = max_log_likelihood_instance.gaussian.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_exponential = max_log_likelihood_instance.exponential.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D Gaussian + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

# %%
"""
All methods which give instances give us the same instance of a CollectionPriorModel:
"""

# %%
print(samples.median_pdf_instance)
print(samples.instance_at_upper_sigma)
print(samples.instance_at_lower_sigma)
print(samples.error_instance_at_upper_sigma)
print(samples.error_instance_at_lower_sigma)
print(samples.instance_from_sample_index(sample_index=500))

# %%
"""
So that is that - adding model complexity doesn't change a whole lot about the Result object, other than the switch
to CollectionPriorModels meaning that our instances now have named entries.

The take home point should be that when you name your model components, you should make sure to give them descriptive
and information names that make the use of a result object clear and intuitive!
"""
