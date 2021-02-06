"""
__Example: Result__

In this example, we'll repeat the fit of 1D data of a `Gaussian` profile with a 1D `Gaussian` model using the non-linear
search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files `example/model.py`,`example/analysis.py` and `example/fit.py` to
see how the fit is performed by the code below. The first section of code below is simmply repeating the commands in
`example/fit.py`, so feel free to skip over it until you his the `Result`'s section.
"""
#%matplotlib inline

import autofit as af
import model as m
import analysis as a

from os import path
import matplotlib.pyplot as plt
import numpy as np

"""
__Data__

First, lets load data of a 1D Gaussian, by loading it from a .json file in the directory 
`autofit_workspace/dataset/`, which  simulates the noisy data we fit (check it out to see how we simulate the 
data).
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian. In model.py, you will have noted
this `Gaussian` has 3 parameters (centre, intensity and sigma). These are the free parameters of our model that the
non-linear search fits for, meaning the non-linear parameter space has dimensionality = 3.
"""
model = af.PriorModel(m.Gaussian)

"""
Checkout `autofit_workspace/config/priors` - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We now set up our Analysis, using the class described in `analysis.py`. The analysis describes how given an instance
of our model (a Gaussian) we fit the data and return a log likelihood value. For this simple example, we only have to
pass it the data and its noise-map.
"""
analysis = a.Analysis(data=data, noise_map=noise_map)

"""Returns the non-linear object for emcee and perform the fit."""
emcee = af.Emcee(
    nwalkers=30,
    nsteps=1000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlation_check_for_convergence=True,
    auto_correlation_check_size=100,
    auto_correlation_required_length=50,
    auto_correlation_change_threshold=0.01,
    number_of_cores=1,
)

result = emcee.fit(model=model, analysis=analysis)

"""
__Result__

Here, we'll look in detail at what information is contained in the result.

It contains a `Samples` object, which contains information on the non-linear sampling, for example the parameters. 
The parameters are stored as a list of lists, where the first entry corresponds to the sample index and second entry
the parameter index.
"""
samples = result.samples

print("Final 10 Parameters:")
print(samples.parameters[-10:])

print("Sample 10`s third parameter value (Gaussian -> sigma)")
print(samples.parameters[9][2], "\n")

"""
The Samples class also contains the log likelihood, log prior, log posterior and weights of every accepted sample, 
where:

   - The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
     normalized).

   - The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log
     posterior value.

   - The log posterior is log_likelihood + log_prior.

   - The weight gives information on how samples should be combined to estimate the posterior. The weight values 
     depend on the sampler used, for MCMC samples they are all 1 (e.g. all weighted equally).
     
Lets inspect the last 10 values of each for the analysis.     
"""
print("Final 10 Log Likelihoods:")
print(samples.log_likelihoods[-10:])

print("Final 10 Log Priors:")
print(samples.log_priors[-10:])

print("Final 10 Log Posteriors:")
print(samples.log_posteriors[-10:])

print("Final 10 Sample Weights:")
print(samples.weights[-10:], "\n")

"""
The median pdf vector is readily available from the `Samples` object for you convenience (and if a nested sampling
`NonLinearSearch` is used instead, it will use an appropriate method to estimate the parameters):
"""
median_pdf_vector = samples.median_pdf_vector
print("Median PDF Vector:")
print(median_pdf_vector, "\n")

"""
The samples contain many useful vectors, including the samples with the highest likelihood and posterior values:
"""
max_log_likelihood_vector = samples.max_log_likelihood_vector
max_log_posterior_vector = samples.max_log_posterior_vector

print("Maximum Log Likelihood Vector:")
print(max_log_likelihood_vector)

print("Maximum Log Posterior Vector:")
print(max_log_posterior_vector, "\n")

"""
It also provides methods for computing the error estimates of all parameters at an input sigma confidence limit, which
can be returned at the values of the parameters including their errors or the size of the errors on each parameter:
"""
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

"""
Results vectors return the results as a list, which means you need to know the parameter ordering. The list of
parameter names are available as a property of the `Samples`, as are parameter labels which can be used for labeling
figures:
"""
print(samples.model.model_component_and_parameter_names)
print(samples.model.parameter_labels)
print("\n")

"""
Results can instead be returned as an instance, which is an instance of the model using the Python classes used to
compose it:
"""
max_log_likelihood_instance = samples.max_log_likelihood_instance

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", max_log_likelihood_instance.centre)
print("Intensity = ", max_log_likelihood_instance.intensity)
print("Sigma = ", max_log_likelihood_instance.sigma, "\n")

"""
For our example problem of fitting a 1D `Gaussian` profile, this makes it straight forward to plot the maximum
likelihood model:
"""
model_data = samples.max_log_likelihood_instance.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D `Gaussian` profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""All methods above are available as an instance:"""
median_pdf_instance = samples.median_pdf_instance
instance_at_upper_sigma = samples.instance_at_upper_sigma
instance_at_lower_sigma = samples.instance_at_lower_sigma
error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

"""An instance of any accepted sample can be created:"""
instance = samples.instance_from_sample_index(sample_index=500)

print("Gaussian Instance of sample 5000:")
print("Centre = ", instance.centre)
print("Intensity = ", instance.intensity)
print("Sigma = ", instance.sigma, "\n")

"""
If a nested sampling `NonLinearSearch` is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:
"""
log_evidence = samples.log_evidence

"""
At this point, you might be wondering what else the results contains - pretty much everything we discussed above was a
part of its *samples* property! For projects which use **PyAutoFit**'s phase API (see the howtofit tutrials), the 
`Result`'s object can be extended to include model-specific results.

For example, we may extend the results of our 1D `Gaussian` example to include properties containing the maximum
log likelihood of the summed model data and for every individual profile in the model.

(The commented out functions below are llustrative of the API we can create, but do not work in this example given we 
are not using the phase API.)
"""
# max_log_likelihood_profile = results.max_log_likelihood_profile
# max_log_likelihood_profile_list = results.max_log_likelihood_profile_list
