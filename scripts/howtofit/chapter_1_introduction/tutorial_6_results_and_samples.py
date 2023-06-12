"""
Tutorial 6: Results And Samples
===============================

In this tutorial, we'll cover all of the output that comes from a non-linear search's `Result`  object.

We used this object at various points in the chapter. The bulk of material covered here is described in the example
script `autofit_workspace/overview/simple/result.py`. Nevertheless, it is a good idea to refresh ourselves about how
results in **PyAutoFit** work before covering more advanced material.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

"""
__Data__

Load the dataset from the `autofit_workspace/dataset` folder.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Reused Functions__

We'll reuse the `plot_profile_1d` and `Analysis` classes of the previous tutorial.
"""


def plot_profile_1d(
    xvalues,
    profile_1d,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
):
    plt.errorbar(
        x=xvalues,
        y=profile_1d,
        yerr=errors,
        color=color,
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    if not path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """
        xvalues = np.arange(self.data.shape[0])

        return sum(
            [
                profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
                for profile in instance
            ]
        )

    def visualize(self, paths, instance, during_analysis):
        """
        This method is identical to the previous tutorial, except it now uses the `model_data_from_instance` method
        to create the profile.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0

        """The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`)."""
        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )


"""
__Model Fit__

Now lets run the non-linear search to get ourselves a `Result`.
"""
import profiles as p

model = af.Collection(
    gaussian=af.Model(p.Gaussian), exponential=af.Model(p.Exponential)
)

analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    name="tutorial_6_results_and_samples",
    path_prefix=path.join("howtofit", "chapter_1"),
)

print(
    """
    Emcee has begun running.
    Checkout the autofit_workspace/output/howtofit/tutorial_6__results_and_samples
    folder for live output of the results.
    This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

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
print("Centre = ", max_lh_instance.gaussian.centre)
print("Normalization = ", max_lh_instance.gaussian.normalization)
print("Sigma = ", max_lh_instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", max_lh_instance.exponential.centre)
print("Normalization = ", max_lh_instance.exponential.normalization)
print("Sigma = ", max_lh_instance.exponential.rate, "\n")

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

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", median_pdf_instance.gaussian.centre)
print("Normalization = ", median_pdf_instance.gaussian.normalization)
print("Sigma = ", median_pdf_instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", median_pdf_instance.exponential.centre)
print("Normalization = ", median_pdf_instance.exponential.normalization)
print("Sigma = ", median_pdf_instance.exponential.rate, "\n")

"""
__Plot__

Because results are returned as instances, it is straight forward to use them and their associated functionality
to make plots of the results:
"""
model_gaussian = max_lh_instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = max_lh_instance.exponential.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D `Gaussian` + Exponential profile data.")
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

print("Upper Error values of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", errors_at_upper_sigma_instance.gaussian.centre)
print("Normalization = ", errors_at_upper_sigma_instance.gaussian.normalization)
print("Sigma = ", errors_at_upper_sigma_instance.gaussian.sigma, "\n")

print("lower Error values of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", errors_at_lower_sigma_instance.gaussian.centre)
print("Normalization = ", errors_at_lower_sigma_instance.gaussian.normalization)
print("Sigma = ", errors_at_lower_sigma_instance.gaussian.sigma, "\n")

"""
They can also be returned at the values of the parameters at their error values:
"""
values_at_upper_sigma_instance = samples.values_at_upper_sigma(sigma=3.0)
values_at_lower_sigma_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", values_at_upper_sigma_instance.gaussian.centre)
print("Normalization = ", values_at_upper_sigma_instance.gaussian.normalization)
print("Sigma = ", values_at_upper_sigma_instance.gaussian.sigma, "\n")

print("lower Parameter values w/ errors of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", values_at_lower_sigma_instance.gaussian.centre)
print("Normalization = ", values_at_lower_sigma_instance.gaussian.normalization)
print("Sigma = ", values_at_lower_sigma_instance.gaussian.sigma, "\n")

"""
__PDF__

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
print("Centre = ", max_log_posterior_instance.gaussian.centre)
print("Normalization = ", max_log_posterior_instance.gaussian.normalization)
print("Sigma = ", max_log_posterior_instance.gaussian.sigma, "\n")


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

We can create an instance of any lens model -- below we create an instance of the last accepted model.
"""
instance = samples.from_sample_index(sample_index=-1)

print("Gaussian Instance of last sample")
print("Centre = ", instance.gaussian.centre)
print("Normalization = ", instance.gaussian.normalization)
print("Sigma = ", instance.gaussian.sigma, "\n")

"""
__Bayesian Evidence__

If a nested sampling `NonLinearSearch` is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:
"""
log_evidence = samples.log_evidence

"""
__Derived Errors (PDF from samples)__

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

    sigma = instance.gaussian.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, upper_fwhm, lower_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"FWHM = {median_fwhm} ({upper_fwhm} {lower_fwhm}")

"""
__Samples Filtering__

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
Above, we specified each path as a list of tuples of strings. 

This is how the PyAutoFit source code stores the path to different components of the model, but it is not 
in-profile_1d with the PyAutoFIT API used to compose a model.

We can alternatively use the following API:
"""
samples = result.samples

samples = samples.with_paths(["gaussian.centre"])

print("All parameters of the very first sample (containing only the Gaussian centre).")
print(samples.parameter_lists[0])

"""
Above, we filtered the `Samples` but asking for all parameters which included the path ("gaussian", "centre").

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
Finish.
"""
