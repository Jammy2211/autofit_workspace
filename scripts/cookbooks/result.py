"""
Cookbook: Results
=================

After a non-linear search has completed, it returns a `Result` object that contains information on fit, such as
the maximum likelihood model instance, the errors on each parameter and the Bayesian evidence.

This cookbook provides an overview of using the results.

__Contents__

 - Model Fit: Perform a simple model-fit to create a `Result` object.
 - Info: Print the `info` attribute of the `Result` object to display a summary of the model-fit.
 - Loading From Hard-disk: Loading results from hard-disk to Python variables via the aggregator.
 - Samples: The `Samples` object contained in the `Result`, containing all non-linear samples (e.g. parameters,
   log likelihoods, etc.).
 - Instances: Returning instances of the model corresponding to a particular sample (e.g. the maximum log likelihood).
 - Posterior / PDF: The median PDF model instance and PDF vectors of all model parameters via 1D marginalization.
 - Errors: The errors on every parameter estimated from the PDF, computed via marginalized 1D PDFs at an input sigma.
 - Samples Summary: A summary of the samples of the non-linear search (e.g. the maximum log likelihood model) which can
   be faster to load than the full set of samples.
 - Sample Instance: The model instance of any accepted sample.
 - Search Plots: Plots of the non-linear search, for example a corner plot or 1D PDF of every parameter.
 - Maximum Likelihood: The maximum log likelihood model value.
 - Bayesian Evidence: The log evidence estimated via a nested sampling algorithm.
 - Collection: Results created from models defined via a `Collection` object.
 - Lists: Extracting results as Python lists instead of instances.
 - Latex: Producing latex tables of results (e.g. for a paper).

The following sections outline how to use advanced features of the results, which you may skip on a first read:

 - Derived Quantities: Computing quantities and errors for quantities and parameters not included directly in the model.
 - Result Extension: Extend the `Result` object with new attributes and methods (e.g. `max_log_likelihood_model_data`).
 - Samples Filtering: Filter the `Samples` object to only contain samples fulfilling certain criteria.
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
__Model Fit__

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

We do this below using the standard API and noisy 1D signal example, which you should be familiar with from other 
example scripts.

Note that the `Gaussian` and `Analysis` classes come via the `af.ex` module, which contains example model components
that are identical to those found throughout the examples.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    name="cookbook_result",
    nwalkers=30,
    nsteps=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

Printing the `info` attribute shows the overall result of the model-fit in a human readable format.
"""
print(result.info)

"""
__Loading From Hard-disk__

When performing fits which output results to hard-disk, a `files` folder is created containing .json / .csv files of 
the model, samples, search, etc. You should check it out now for a completed fit on your hard-disk if you have
not already!

These files can be loaded from hard-disk to Python variables via the aggregator, making them accessible in a 
Python script or Jupyter notebook. They are loaded as the internal **PyAutoFit** objects we are familiar with,
for example the `model` is loaded as the `Model` object we passed to the search above.

Below, we will access these results using the aggregator's `values` method. A full list of what can be loaded is
as follows:

 - `model`: The `model` defined above and used in the model-fit (`model.json`).
 - `search`: The non-linear search settings (`search.json`).
 - `samples`: The non-linear search samples (`samples.csv`).
 - `samples_info`: Additional information about the samples (`samples_info.json`).
 - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
 - `info`: The info dictionary passed to the search (`info.json`).
 - `covariance`: The inferred covariance matrix (`covariance.csv`).
 - `data`: The 1D noisy data used that is fitted (`data.json`).
 - `noise_map`: The 1D noise-map fitted (`noise_map.json`).
 
The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does not reperform calculations using the full 
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the `samples.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "cookbook_result"),
)

"""
__Generators__

Before using the aggregator to inspect results, lets discuss Python generators. 

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects 
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory 
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once. 
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the 
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the 
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the `values` method. This takes the `name` of the
object we want to create a generator of, for example inputting `name=samples` will return the results `Samples`
object (which is illustrated in detail below).
"""
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

"""
__Samples__

The result contains a `Samples` object, which contains all samples of the non-linear search.

Each sample corresponds to a set of model parameters that were evaluated and accepted by the non linear search, 
in this example `emcee.` 

This includes their log likelihoods, which are used for computing additional information about the model-fit,
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

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, where:

- The `log_likelihood` is the value evaluated in the `log_likelihood_function`.

- The `log_prior` encodes information on how parameter priors map log likelihood values to log posterior values.

- The `log_posterior` is `log_likelihood + log_prior`.

- The `weight` gives information on how samples are combined to estimate the posterior, which depends on type of search
  used (for `Emcee` they are all 1's meaning they are weighted equally).

Lets inspect these values for the tenth sample.
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Instances__

Many results can be returned as an instance of the model, using the Python class structure of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.

The attributes of the `instance` (`centre`, `normalization` and `sigma`) have these names due to how we composed 
the `Gaussian` class via the `Model` above. They would be named structured and named differently if we hd 
used a `Collection` and different names.
"""
instance = samples.max_log_likelihood()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", instance.centre)
print("Normalization = ", instance.normalization)
print("Sigma = ", instance.sigma, "\n")

"""
This makes it straight forward to plot the median PDF model:
"""
model_data = instance.model_data_1d_via_xvalues_from(xvalues=np.arange(data.shape[0]))

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D `Gaussian` profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Posterior / PDF__

The result contains the full posterior information of our non-linear search, which can be used for parameter 
estimation. 

PDF stands for "Probability Density Function" and it quantifies probability of each model parameter having values
that are sampled. It therefore enables error estimation via a process called marginalization.

The median pdf vector is available, which estimates every parameter via 1D marginalization of their PDFs.
"""
instance = samples.median_pdf()

print("Median PDF `Gaussian` Instance:")
print("Centre = ", instance.centre)
print("Normalization = ", instance.normalization)
print("Sigma = ", instance.sigma, "\n")

"""
__Errors__

Methods for computing error estimates on all parameters are provided. 

This again uses 1D marginalization, now at an input sigma confidence limit. 

By inputting `sigma=3.0` margnialization find the values spanning 99.7% of 1D PDF. Changing this to `sigma=1.0`
would give the errors at the 68.3% confidence limit.
"""
instance_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0)
instance_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0)

print("Upper Error values (at 3.0 sigma confidence):")
print("Centre = ", instance_upper_sigma.centre)
print("Normalization = ", instance_upper_sigma.normalization)
print("Sigma = ", instance_upper_sigma.sigma, "\n")

print("lower Error values (at 3.0 sigma confidence):")
print("Centre = ", instance_lower_sigma.centre)
print("Normalization = ", instance_lower_sigma.normalization)
print("Sigma = ", instance_lower_sigma.sigma, "\n")

"""
They can also be returned at the values of the parameters at their error values.
"""
instance_upper_values = samples.values_at_upper_sigma(sigma=3.0)
instance_lower_values = samples.values_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
print("Centre = ", instance_upper_values.centre)
print("Normalization = ", instance_upper_values.normalization)
print("Sigma = ", instance_upper_values.sigma, "\n")

print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
print("Centre = ", instance_lower_values.centre)
print("Normalization = ", instance_lower_values.normalization)
print("Sigma = ", instance_lower_values.sigma, "\n")

"""
__Samples Summary__

The samples summary contains a subset of results access via the `Samples`, for example the maximum likelihood model
and parameter error estimates.

Using the samples method above can be slow, as the quantities have to be computed from all non-linear search samples
(e.g. computing errors requires that all samples are marginalized over). This information is stored directly in the
samples summary and can therefore be accessed instantly.
"""
print(samples.summary().max_log_likelihood_sample)

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
__Search Plots__

The Probability Density Functions (PDF's) of the results can be plotted using the non-linear search in-built 
visualization tools.

This fit used `Emcee` therefore we use the `MCMCPlotter` for visualization, which wraps the Python library `corner.py`.

The `autofit_workspace/*/plots` folder illustrates other packages that can be used to make these plots using
the standard output results formats (e.g. `GetDist.py`).
"""
plotter = aplt.MCMCPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Maximum Likelihood__

The maximum log likelihood value of the model-fit can be estimated by simple taking the maximum of all log
likelihoods of the samples.

If different models are fitted to the same dataset, this value can be compared to determine which model provides
the best fit (e.g. which model has the highest maximum likelihood)?
"""
print("Maximum Log Likelihood: \n")
print(max(samples.log_likelihood_list))

"""
__Bayesian Evidence__

If a nested sampling non-linear search is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).

A full discussion of Bayesian model comparison is given in `autofit_workspace/*/features/bayes_model_comparison.py`.
"""
log_evidence = samples.log_evidence
print(f"Log Evidence: {log_evidence}")

"""
__Collection__

The examples correspond to a model where `af.Model(Gaussian)` was used to compose the model.

Below, we illustrate how the results API slightly changes if we compose our model using a `Collection`:
"""
model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    nwalkers=50,
    nsteps=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
The `result.info` shows the result for the model with both a `Gaussian` and `Exponential` profile.
"""
print(result.info)

"""
Result instances again use the Python classes used to compose the model. 

However, because our fit uses a `Collection` the `instance` has attributes named according to the names given to the
`Collection`, which above were `gaussian` and `exponential`.

For complex models, with a large number of model components and parameters, this offers a readable API to interpret
the results.
"""
samples = result.samples

instance = samples.max_log_likelihood()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", instance.gaussian.centre)
print("Normalization = ", instance.gaussian.normalization)
print("Sigma = ", instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", instance.exponential.centre)
print("Normalization = ", instance.exponential.normalization)
print("Sigma = ", instance.exponential.rate, "\n")

"""
__Lists__

All results can alternatively be returned as a 1D list of values, by passing `as_instance=False`:
"""
max_lh_list = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_list, "\n\n")

"""
The list above does not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:

 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models they are important for tracking the parameters of the model.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
All the methods above are available as lists.
"""
instance = samples.median_pdf(as_instance=False)
values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)

"""
__Latex__

If you are writing modeling results up in a paper, you can use inbuilt latex tools to create latex table code which 
you can copy to your .tex document.

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
__Derived Quantities__

The parameters `centre`, `normalization` and `sigma` are the model parameters of the `Gaussian`. They are sampled
directly by the non-linear search and we can therefore use the `Samples` object to easily determine their values and 
errors.

Derived quantities (also called latent variables) are those which are not sampled directly by the non-linear search, 
but one may still wish to know their values and errors after the fit is complete. For example, what if we want the 
error on the full width half maximum (FWHM) of the Gaussian? 

This is achieved by adding each derived quantity to the `Model` objects with the `@derived_quantity` decorator, with
an example for the `Gaussian` model component given below:
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization: float = 0.1,  # <- are the Gaussian`s model parameters.
        sigma: float = 0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    @af.derived_quantity
    def fwhm(self):
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma


"""
The fit performed above used a `Gaussian` object a `fwhm` `derived_property`. 

This leads to a number of noteworthy outputs:

 - A `model.derived` file is output to the results folder, which includes the value and error of all derived quantities 
   based on the non-linear search samples (in this example only the `fwhm`).
   
 - A `derived_quantities.csv` is output which lists every accepted sample's value of every derived quantity, which is again
   analogous to the `samples.csv` file (in this example only the `fwhm`). 
     
 - A `derived_summary.json` is output which acts analogously to `samples_summary.json` but for the derived quantities 
   of the model (in this example only the `fwhm`).

Derived quantities are also accessible via the `Samples` object, following a similar API to the model parameters:
"""
instance = samples.max_log_likelihood()

print(f"Max Likelihood FWHM: {instance.fwhm}")

instance = samples.median_pdf()

print(f"Median PDF FWHM {instance.fwhm}")

"""
The summary of the derived quantities can be output using the `derived_quantities_summary_dict` method:
"""
print(samples.derived_quantities_summary_dict())

"""
__Derived Errors Manual (Advanced)__

The derived quantities decorator above provides a simple interface for computing the errors of a derived quantity and
ensuring all results are easily inspected in the output results folder.

However, you may wish to compute the errors of a derived quantity manually. For example, if it is a quantity that 
you did not decorate before performing the fit, or if it is computationally expensive to compute and you only want
to compute it specific circumstances.

Below, we create the PDF of the derived quantity, the FWHM, manually, which we marginalize over using the same function 
we use to marginalize model parameters. We compute the FWHM of every accepted model sampled by the non-linear search 
and use this determine the PDF of the FWHM. 

When combining the FWHM's we weight each value by its `weight`. For Emcee, an MCMC algorithm, the weight of every 
sample is 1, but weights may take different values for other non-linear searches.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the FWHM to compute 
its error, we also pass the weight list of the samples.
"""
fwhm_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model)

    sigma = instance.gaussian.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, lower_fwhm, upper_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"FWHM = {median_fwhm} ({upper_fwhm} {lower_fwhm}")

"""
The calculation above could be computationally expensive, if there are many samples and the derived quantity is
slow to compute.

An alternative approach, which will provide comparable accuracy provided enough draws are used, is to sample 
points randomy from the PDF of the model and use these to compute the derived quantity.

Draws are from the PDF of the model, so the weights of the samples are accounted for and we therefore do not
pass them to the `marginalize` function (it essentially treats all samples as having equal weight).
"""
random_draws = 50

fwhm_list = []

for i in range(random_draws):
    instance = samples.draw_randomly_via_pdf()

    sigma = instance.gaussian.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, lower_fwhm, upper_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"fwhm = {median_fwhm} ({upper_fwhm} {lower_fwhm}")

"""
__Result Extensions (Advanced)__

You might be wondering what else the results contains, as nearly everything we discussed above was a part of its 
`samples` property! The answer is, not much, however the result can be extended to include  model-specific results for 
your project. 

We detail how to do this in analysis cookbook, but for the example of fitting a 1D Gaussian we could extend
the result to include the maximum log likelihood profile:

(The commented out functions below are llustrative of the API we can create by extending a result).
"""
# max_log_likelihood_profile = results.max_log_likelihood_profile

"""
__Samples Filtering (Advanced)__

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.with_paths([("gaussian", "centre")])

print("All parameters of the very first sample (containing only the Gaussian centre.")
print(samples.parameter_lists[0])

print("Maximum Log Likelihood Model Instances (containing only the Gaussian centre):\n")
print(samples.max_log_likelihood(as_instance=False))

"""
We specified each path as a list of tuples of strings. 

This is how the source code internally stores the path to different components of the model, but it is not 
consistent with the API used to compose a model.

We can alternatively use the following API:
"""
samples = result.samples

samples = samples.with_paths(["gaussian.centre"])

print("All parameters of the very first sample (containing only the Gaussian centre).")
print(samples.parameter_lists[0])

"""
We filtered the `Samples` above by asking for all parameters which included the path ("gaussian", "centre").

We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the Gaussian's `centre` to be left with 2 parameters; the `normalization` and `sigma`.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.without_paths(["gaussian.centre"])

print(
    "All parameters of the very first sample (containing only the Gaussian normalization and sigma)."
)
print(samples.parameter_lists[0])

"""
__Wrap Up__

Adding model complexity does not change the behaviour of the Result object, other than the switch
to Collections meaning that our instances now have named entries.

When you name your model components, you should make sure to give them descriptive and information names that make 
the use of a result object clear and intuitive!

__Database__

For large-scaling model-fitting problems to large datasets, the results of the many model-fits performed can be output
and stored in a queryable sqlite3 database. The `Result` and `Samples` objects have been designed to streamline the 
analysis and interpretation of model-fits to large datasets using the database.

Checkout the database cookbook for more details on how to use the database.
"""
