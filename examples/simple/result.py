from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np

from autoconf import conf
import autofit as af
from autofit_workspace.examples.simple import model as m
from autofit_workspace.examples.simple import analysis as a

"""
__Example: Result__

In this example, we'll repeat the fit of 1D data of a Gaussian profile with a 1D Gaussian model using the non-linear 
search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files 'example/model.py','example/analysis.py' and 'example/fit.py' to 
see how the fit is performed by the code below. The first section of code below is simmply repeating the commands in
'example/fit.py', so feel free to skip over it until you his the __Result__ section.
"""

# %%
"""
__Paths__

Setup the path to the autofit_workspace, using a relative directory name.
"""

# %%
#%matplotlib inline

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

First, lets load our data of a 1D Gaussian.
"""

# %%
dataset_path = f"{workspace_path}/dataset/gaussian_x1"

data_hdu_list = fits.open(f"{dataset_path}/data.fits")
data = np.array(data_hdu_list[0].data)

noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
noise_map = np.array(noise_map_hdu_list[0].data)

# %%
"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian. In model.py, you will have noted
this Gaussian has 3 parameters (centre, intensity and sigma). These are the free parameters of our model that the
non-linear search fits for, meaning the non-linear parameter space has dimensionality = 3.
"""

# %%
model = af.PriorModel(m.Gaussian)

# %%
"""
Checkout 'autofit_workspace/config/json_priors' - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the non-linear search as shown below.
"""

# %%
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

# %%
"""
__Analysis__

We now set up our Analysis, using the class described in 'analysis.py'. The analysis describes how given an instance
of our model (a Gaussian) we fit the data and return a log likelihood value. For this simple example, we only have to
pass it the data and its noise-map.
"""

# %%
analysis = a.Analysis(data=data, noise_map=noise_map)

# %%
"""
Create the non-linear object for emcee and perform the fit.
"""

# %%
emcee = af.Emcee(
    nwalkers=30,
    nsteps=1000,
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

Here, we'll look in detail at what information is contained in the result.

It contains a *Samples* object, which contains information on the non-linear sampling, for example the parameters. 
The parameters are stored as a a list of lists, where the first entry corresponds to the sample index and second entry
the parameter index.
"""

# %%
samples = result.samples
print("All Parameters:")
print(samples.parameters)
print("Sample 10's third parameter value (Gaussian -> sigma)")
print(samples.parameters[9][1], "\n")

# %%
"""
The parameters are a list of lists of all accepted parameter values sampled by the non-linear search. Also available
are lists of the likelihood, prior, posterior and weight values associated with every sample:
"""

# %%
print("All Log Likelihoods:")
print(samples.log_likelihoods)
print("All Log Priors:")
print(samples.log_priors)
print("All Log Posteriors:")
print(samples.log_posteriors)
print("All Sample Weights:")
print(samples.weights, "\n")

# %%
"""
For MCMC analysis, these can be used perform parameter estimation by binning the samples in a histogram (assuming we
have removed the burn-in phase):
"""

# %%
samples_after_burn_in = result.samples.samples_after_burn_in

median_pdf_vector = [
    float(np.percentile(samples_after_burn_in[:, i], [50]))
    for i in range(model.prior_count)
]

# %%
"""
The most probable vector is readily available from the *Samples* object for you convenience (and if a nested sampling
*non-linear search* is used instead, it will use an appropriate method to estimate the parameters):
"""

# %%
median_pdf_vector = samples.median_pdf_vector
print("Most Probable Vector:")
print(median_pdf_vector, "\n")

# %%
"""
The samples contain many useful vectors, including the samples with the highest likelihood and posterior values:
"""

# %%
max_log_likelihood_vector = samples.max_log_likelihood_vector
max_log_posterior_vector = samples.max_log_posterior_vector

print("Maximum Log Likelihood Vector:")
print(max_log_likelihood_vector)
print("Maximum Log Posterior Vector:")
print(max_log_posterior_vector, "\n")


# %%
"""
It also provides methods for computing the error estimates of all parameters at an input sigma confidence limit, which
can be returned at the values of the parameters including their errors or the size of the errors on each parameter:
"""

# %%
vector_at_upper_sigma = samples.vector_at_upper_sigma(sigma=3.0)
vector_at_lower_sigma = samples.vector_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
print(vector_at_upper_sigma)
print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
print(vector_at_lower_sigma, "\n")

error_vector_at_upper_sigma = samples.error_vector_at_upper_sigma(sigma=3.0)
error_vector_at_lower_sigma = samples.error_vector_at_lower_sigma(sigma=3.0)

print("Upper Error values (at 3.0 sigma confidence):")
print(error_vector_at_upper_sigma)
print("lower Error values (at 3.0 sigma confidence):")
print(error_vector_at_lower_sigma, "\n")

# %%
"""
Results vectors return the results as a list, which means you need to know the parameter ordering. The list of
parameter names are available as a property of the *Samples*, as are parameter labels which can be used for labeling
figures:
"""

# %%
print(samples.parameter_names)
print(samples.parameter_labels)
print("\n")

# %%
"""
Results can instead be returned as an instance, which is an instance of the model using the Python classes used to
compose it:
"""

# %%
max_log_likelihood_instance = samples.max_log_likelihood_instance

print("Max Log Likelihood Gaussian Instance:")
print("Centre = ", max_log_likelihood_instance.centre)
print("Intensity = ", max_log_likelihood_instance.intensity)
print("Sigma = ", max_log_likelihood_instance.sigma, "\n")

# %%
"""
For our example problem of fitting a 1D Gaussian profile, this makes it straight forward to plot the maximum
likelihood model:
"""

# %%
model_data = samples.max_log_likelihood_instance.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D Gaussian profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

# %%
"""
All methods above are available as an instance:
"""

# %%
median_pdf_instance = samples.median_pdf_instance
instance_at_upper_sigma = samples.instance_at_upper_sigma
instance_at_lower_sigma = samples.instance_at_lower_sigma
error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

# %%
"""
An instance of any accepted sample can be created:
"""

# %%
instance = samples.instance_from_sample_index(sample_index=500)
print("Gaussian Instance of sample 5000:")
print("Centre = ", instance.centre)
print("Intensity = ", instance.intensity)
print("Sigma = ", instance.sigma, "\n")

# %%
"""
If a nested sampling *non-linear search* is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:
"""

# %%
log_evidence = samples.log_evidence

# %%
"""
At this point, you might be wondering what else the results contains - pretty much everything we discussed above was a
part of its *samples* property! For projects which use **PyAutoFit**'s phase API (see the howtofit tutrials), the 
*Results* object can be extended to include model-specific results.

For example, we may extend the results of our 1D Gaussian example to include properties containing the maximum
log likelihood of the summed model data and for every individual profile in the model.

(The commented out functions below are llustrative of the API we can create, but do not work in this example given we 
are not using the phase API.)
"""

# %%
# max_log_likelihood_line = results.max_log_likelihood_line
# max_log_likelihood_line_list = results.max_log_likelihood_line_list
