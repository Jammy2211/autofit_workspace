"""
__Example: Result__

In this example, we'll repeat the fit of 1D data of a `Gaussian` profile with a 1D `Gaussian` model using the non-linear
search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files `example/model.py`,`example/analysis.py` and `example/fit.py` to
see how the fit is performed by the code below. The first section of code below is simmply repeating the commands in
`example/fit.py`, so feel free to skip over it until you his the `Result`'s section.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt

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

Next, we create our model, which in this case corresponds to a single Gaussian. 

We use the `af.ex` package to create the `Gaussian`, where `af.ex` stands for **PyAutoFit** example objects used
in example scripts such as this one. 

The `Gaussian` model below is  identical 1D `Gaussian` to the one used in the `fit.py` example script.
"""
model = af.Model(af.ex.Gaussian)

"""
Checkout `autofit_workspace/config/priors` - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We now set up our `Analysis`, which describes how given an instance of our model (a Gaussian) we fit the data and 
return a log likelihood value. 

We use the `af.ex` package to create the `Analysis`, where `af.ex` stands for **PyAutoFit** example objects used
in example scripts such as this one. 

The `Analysis` below is an identical to the one used in the `fit.py` example script.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
We now create the `Emcee` non-linear search and perform a model-fit to get the result.
"""
search = af.Emcee(
    nwalkers=30,
    nsteps=1000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

Here, we'll look in detail at what information is contained in the `Result`.

It contains an `info` attribute which prints the result in readable format.
"""
print(result.info)

"""
__Samples__

The result contains a `Samples` object, which contains all of the non-linear search samples. 

Each sample corresponds to a set of model parameters that were evaluated and accepted by our non linear search, 
in this example emcee. 

This also includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the MCMC algorithm Emcee, so the `Samples` object returned is a `SamplesMCMC` object.
"""
samples = result.samples

print("MCMC Samples: \n")
print(samples)

"""
__Parameters__

The parameters are stored as a list of lists, where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
samples = result.samples
print("Sample 5's second parameter value (Gaussian -> normalization):")
print(samples.parameter_lists[4][1])
print("Sample 10`s third parameter value (Gaussian -> sigma)")
print(samples.parameter_lists[9][2], "\n")

"""
__Figures of Merit__

The Samples class also contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, 
where:

- The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
normalized).

- The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log 
posterior value.

- The log posterior is log_likelihood + log_prior.

- The weight gives information on how samples should be combined to estimate the posterior. The weight values depend on 
the sampler used, for MCMC samples they are all 1 (e.g. all weighted equally).
     
Lets inspect the last 10 values of each for the analysis.     
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Instances__

The `Samples` contains many results which are returned as an instance of the model, using the Python class structure
of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.
"""
max_lh_instance = samples.max_log_likelihood()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", max_lh_instance.centre)
print("Normalization = ", max_lh_instance.normalization)
print("Sigma = ", max_lh_instance.sigma, "\n")

"""
__Vectors__

All results can alternatively be returned as a 1D vector of values, by passing `as_instance=False`:
"""
max_lh_vector = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_vector, "\n\n")

"""
__Labels__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:
 
 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models illustrated in other tutorials their utility will become clear.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
From here on, we will returned all results information as instances, but every method below can be returned as a
vector via the `as_instance=False` input.

__Posterior / PDF__

The ``Result`` object contains the full posterior information of our non-linear search, which can be used for
parameter estimation. 

The median pdf vector is available from the `Samples` object, which estimates the every parameter via 1D 
marginalization of their PDFs.
"""
median_pdf_instance = samples.median_pdf()

print("Median PDF `Gaussian` Instance:")
print("Centre = ", median_pdf_instance.centre)
print("Normalization = ", median_pdf_instance.normalization)
print("Sigma = ", median_pdf_instance.sigma, "\n")

"""
For our example problem of fitting a 1D `Gaussian` profile, this makes it straight forward to plot the maximum
likelihood model:
"""
model_data = samples.max_log_likelihood().model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D `Gaussian` profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Errors__

The samples include methods for computing the error estimates of all parameters, via 1D marginalization at an 
input sigma confidence limit. 
"""
errors_at_upper_sigma_instance = samples.errors_at_upper_sigma(sigma=3.0)
errors_at_lower_sigma_instance = samples.errors_at_lower_sigma(sigma=3.0)

print("Upper Error values (at 3.0 sigma confidence):")
print("Centre = ", errors_at_upper_sigma_instance.centre)
print("Normalization = ", errors_at_upper_sigma_instance.normalization)
print("Sigma = ", errors_at_upper_sigma_instance.sigma, "\n")

print("lower Error values (at 3.0 sigma confidence):")
print("Centre = ", errors_at_lower_sigma_instance.centre)
print("Normalization = ", errors_at_lower_sigma_instance.normalization)
print("Sigma = ", errors_at_lower_sigma_instance.sigma, "\n")

"""
They can also be returned at the values of the parameters at their error values:
"""
values_at_upper_sigma_instance = samples.values_at_upper_sigma(sigma=3.0)
values_at_lower_sigma_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
print("Centre = ", values_at_upper_sigma_instance.centre)
print("Normalization = ", values_at_upper_sigma_instance.normalization)
print("Sigma = ", values_at_upper_sigma_instance.sigma, "\n")

print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
print("Centre = ", values_at_lower_sigma_instance.centre)
print("Normalization = ", values_at_lower_sigma_instance.normalization)
print("Sigma = ", values_at_lower_sigma_instance.sigma, "\n")

"""
__Plot__

Because results are returned as instances, it is straight forward to use them and their associated functionality
to make plots of the results:
"""
model_data = max_lh_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D `Gaussian` data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search Plots__

The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization 
tool `corner.py`, which is wrapped via the `EmceePlotter` object.
"""
search_plotter = aplt.EmceePlotter(samples=result.samples)
search_plotter.corner()

"""
__Other Results__

The samples contain many useful vectors, including the samples with the highest posterior values.
"""
max_log_posterior_instance = samples.max_log_posterior()

print("Maximum Log Posterior Vector:")
print("Centre = ", max_log_posterior_instance.centre)
print("Normalization = ", max_log_posterior_instance.normalization)
print("Sigma = ", max_log_posterior_instance.sigma, "\n")


"""
All methods above are available as a vector:
"""
median_pdf_instance = samples.median_pdf(as_instance=False)
values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)

"""
__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any model -- below we create an instance of the last accepted model.
"""
instance = samples.from_sample_index(sample_index=-1)

print("Gaussian Instance of last sample")
print("Centre = ", instance.centre)
print("Normalization = ", instance.normalization)
print("Sigma = ", instance.sigma, "\n")

"""
__Bayesian Evidence__

If a nested sampling `NonLinearSearch` is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:
"""
log_evidence = samples.log_evidence

"""
__Derived Errors (Advanced)__

Computing the errors of a quantity like the `sigma` of the Gaussian is simple, because it is sampled by the non-linear 
search. Thus, to get their errors above we used the `Samples` object to simply marginalize over all over parameters 
via the 1D Probability Density Function (PDF).

Computing errors on derived quantitys is more tricky, because it is not sampled directly by the non-linear search. 
For example, what if we want the error on the full width half maximum (FWHM) of the Gaussian? In order to do this
we need to create the PDF of that derived quantity, which we can then marginalize over using the same function we
use to marginalize model parameters.

Below, we compute the FWHM of every accepted model sampled by the non-linear search and use this determine the PDF 
of the FWHM. When combining the FWHM's we weight each value by its `weight`. For Emcee, an MCMC algorithm, the
weight of every sample is 1, but weights may take different values for other non-linear searches.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the FWHM to compute 
its error, we also pass the weight list of the samples.

(Computing the error on the FWHM could be done in much simpler ways than creating its PDF from the list of every
sample. We chose this example for simplicity, in order to show this functionality, which can easily be extended to more
complicated derived quantities.)
"""
fwhm_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model)

    sigma = instance.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, upper_fwhm, lower_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"FWHM = {median_fwhm} ({upper_fwhm} {lower_fwhm}")

"""
__Latex__

If you are writing modeling results up in a paper, you can use PyAutoFit's inbuilt latex tools to create latex table 
code which you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your model
uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.
"""
latex = af.text.Samples.latex(
    samples=result.samples,
    median_pdf_model=True,
    sigma=3.0,
    name_to_label=True,
    include_name=True,
    include_quickmath=True,
    prefix="Example Prefix ",
    suffix=" \\[-2pt]",
)

print(latex)

"""
__Result Extensions (Advanced)__

You might be wondering what else the results contains, as nearly everything we discussed above was a part of its 
`samples` property! The answer is, not much, however the result can be extended to include  model-specific results for 
your project. 

We detail how to do this in the **HowToFit** lectures, but for the example of fitting a 1D Gaussian we could extend
the result to include the maximum log likelihood profile:

(The commented out functions below are llustrative of the API we can create by extending a result).
"""
# max_log_likelihood_profile = results.max_log_likelihood_profile

"""
__Database__

For large-scaling model-fitting problems to large datasets, the results of the many model-fits performed can be output
and stored in a queryable sqlite3 database. The `Result` and `Samples` objects have been designed to streamline the 
analysis and interpretation of model-fits to large datasets using the database.

Checkout `notebooks/features/database.ipynb` for an illustration of using
this tool.
"""
