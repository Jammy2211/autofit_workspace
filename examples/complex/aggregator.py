# %%
"""
__Aggregator Part 1__

After fitting a large suite of data with the same pipeline, the aggregator allows us to load the results and
manipulate / plot them using a Python script or Jupyter notebook.

To begin, we need a set of results that we want to analyse using the aggregator. We'll create this by fitting 3
different data-sets. Each dataset is a single Gaussian and we'll fit them using a single Gaussian model.
"""

# %%
#%matplotlib inline

# %%
from autoconf import conf
import autofit as af

from autofit_workspace.examples.complex import model as m
from autofit_workspace.examples.complex import analysis as a

from astropy.io import fits
import numpy as np
import os

# %%
"""
You need to change the path below to the workspace directory so we can load the dataset
"""

# %%
workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""
Here, for each dataset we are going to set up the correct path, load it, create its mask and fit it with the phase above.

We want our results to be in a folder specific to the dataset. We'll use the dataset_name string to do this. Lets
create a list of all 3 of our dataset names.

We'll also pass these names to the dataset when we create it - the name will be used by the aggregator to name the file
the data is stored. More importantly, the name will be accessible to the aggregator, and we will use it to label 
figures we make via the aggregator.

Note that in this example, we are going to fit datasets that were simulated with different models, 1 a Gaussian + 
Expoenntial (the model we will fit) and two with just a Gaussian (inconsistent with the model we fit).
"""

# %%
dataset_names = ["gaussian_x1__exponential_x1", "gaussian_x1_0", "gaussian_x1_1"]

# %%
"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the dataset) that isn't part of the model-fit is made accessible to the 
aggregator. For example, below we write info on the dataset's data of observation and exposure time.
"""
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

# %%
"""
This for loop runs over every dataset, checkout the comments below for how we set up the path structure.
"""

# %%
"""
__Model__

Next, we create our model, which in this case corresponds to a Gaussian + Exponential. In model.py, you will have
noted the Gaussian has 3 parameters (centre, intensity and sigma) and Exponential 3 parameters (centre, intensity and
rate). These are the free parameters of our model that the non-linear search fits for, meaning the non-linear
parameter space has dimensionality = 6.

In the simple example tutorial, we used a _PriorModel_ to create the model of the Gaussian. PriorModels cannot be used to
compose models from multiple model components and for this example we must instead use the CollectionPriorModel.
"""

# %%
model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

# %%
"""
We can again customize the priors of our model used by the non-linear search.
"""

# %%
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

# %%
for dataset_name in dataset_names:

    # The code below loads the dataset and creates the mask as per usual.

    dataset_path = f"{workspace_path}/dataset/{dataset_name}"

    data_hdu_list = fits.open(f"{dataset_path}/data.fits")
    data = np.array(data_hdu_list[0].data)

    noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
    noise_map = np.array(noise_map_hdu_list[0].data)

    # We use the data and noise-map to create the Analysis class.

    analysis = a.Analysis(data=data, noise_map=noise_map)

    # In all examples so far, our results have gone to the default path, which was the '/output/' folder and a folder
    # name after the non linear search. However, we can manually specify the path of the results in the 'output' folder.
    #
    # We do this belo, using the Paths class and the input parameters 'folders' and 'non_linear_name'. These
    # define the names of folders that the phase goes in, in this case:

    # '/path/to/autofit_workspace/output/aggregator_example/gaussian_x1_0/emcee/'

    emcee = af.Emcee(
        paths=af.Paths(
            folders=["aggregator_example_complex", dataset_name],
            non_linear_name="emcee",
        ),
        nwalkers=30,
        nsteps=1000,
        initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
        auto_correlation_check_for_convergence=True,
        auto_correlation_check_size=100,
        auto_correlation_required_length=50,
        auto_correlation_change_threshold=0.01,
        number_of_cores=1,
    )

    print(
        "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/"
        + dataset_name
        + "/phase_t8"
        "folder for live output of the results."
        "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
    )

    print(
        f"Emcee has begun running - checkout the autofit_workspace/output/{dataset_name} folder for live output of the "
        f"results. This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
    )

    result = emcee.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Checkout the output folder - you should see three new sets of results corresponding to our 3 Gaussian datasets.
Unlike previous tutorials, these folders in the output folder are named after the dataset and contain the folder
with the phase's name, as opposed to just the phase-name folder.

To load these results with the aggregator, we simply point it to the path of the results we want it to inspect.
"""

# %%
output_path = f"{workspace_path}/output"

agg = af.Aggregator(directory=str(output_path))

# %%
"""
To begin, let me quickly explain what a generator is in Python, for those unaware. A generator is an object that 
iterates over a function when it is called. The aggregator creates all objects as generators, rather than lists, or 
dictionaries, or whatever.

Why? Because lists and dictionaries store every entry in memory simultaneously. If you fit many datasets, you'll have 
lots of results and therefore use a lot of memory. This will crash your laptop! On the other hand, a generator only 
stores the object in memory when it runs the function; it is free to overwrite it afterwards. Thus, your laptop won't 
crash!

There are two things to bare in mind with generators:

1) A generator has no length, thus to determine how many entries of data it corresponds to you first must turn it to a 
list.

2) Once we use a generator, we cannot use it again - we'll need to remake it. For this reason, we typically avoid 
   storing the generator as a variable and instead use the aggregator to create them on use.

We can now create a 'samples' generator of every fit. An instance of the Samples class acts as an 
interface between the results of the non-linear fit on your hard-disk and Python.
"""

# %%
samples_gen = agg.values("samples")

# %%
"""
When we print this list of outputs you should see over 3 different MCMCSamples instances. There are more than 3
because the aggregator has loaded the results of previous tutorial as well as the 3 fits we performed above!
"""

# %%
print("Emcee Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(list(samples_gen)), "\n")

# %%
"""
We've encountered the Samples class in previous tutorials, specifically the Result object returned from a phase. In 
this tutorial we'll inspect the Sampels class in more detail. The Samples class contains all the accepted parameter 
samples of the non-linear search, which is a list of lists where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.

"""

# %%
for samples in agg.values("samples"):
    print("All parameters of the very first sample")
    print(samples.parameters[0])
    print("The tenth sample's third parameter")
    print(samples.parameters[9][2])

# %%
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

"""

# %%
for samples in agg.values("samples"):
    print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
    print(samples.log_likelihoods[9])
    print(samples.log_priors[9])
    print(samples.log_posteriors[9])
    print(samples.weights[9])

# %%
"""
To remove the fits of previous tutorials and just keep the MCMCSamples of the 3 datasets fitted in this tutorial we can 
use the aggregator's filter tool. The phase name 'phase_t8' used in this tutorial is unique to all 3 fits, so we can 
use it to filter our results are desired.
"""

# %%
agg_filter = agg.filter(agg.directory.contains("aggregator_example_complex"))
samples_gen = agg_filter.values("samples")

# %%
"""
As expected, this list now has only 3 MCMCSamples, one for each dataset we fitted.
"""

# %%
print("Phase Name Filtered Emcee Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(list(samples_gen)), "\n")

# %%
"""
We can use the samples to create a list of the maximum log likelihood model of each fit to our three images.
"""

# %%
vector = [samps.max_log_likelihood_vector for samps in agg_filter.values("samples")]
print("Maximum Log Likelihood Parameter Lists:\n")
print(vector, "\n")

# %%
"""
This provides us with lists of all model parameters. However, this isn't that much use - which values correspond
to which parameters?

Its more useful to create the maximum log likelihood model instance of every fit.
"""

# %%
instances = [
    samps.max_log_likelihood_instance for samps in agg_filter.values("samples")
]
print("Maximum Log Likelihood Model Instances:\n")
print(instances, "\n")

# %%
"""
A model instance contains all the model components of our fit - which for the fits above was a single gaussian
profile (the word 'gaussian' comes from what we called it in the CollectionPriorModel when making the phase above).
"""

# %%
print(instances[0].gaussian)
print(instances[1].gaussian)
print(instances[2].gaussian)

# %%
"""
This, of course, gives us access to any individual parameter of our maximum log likelihood model. Below, we see that t
he 3 Gaussians were simulated using sigma values of 1.0, 5.0 and 10.0.
"""

# %%
print(instances[0].gaussian.sigma)
print(instances[1].gaussian.sigma)
print(instances[2].gaussian.sigma)

# %%
"""
We can also access the 'most probable' model, which is the model computed by binning all of the accepted Emcee sample
parameters into a histogram, after removing the initial samples where the non-linear sampler is 'burning in' to the 
high likelihood regions of parameter space. 

The median of each 1D histogram (1 for each parameter) is then used to give the median PDF model. This process is 
called 'marginalization' and the hisograms which provide information on the probability estimates of each parameter 
are called the 'Probability Density Function' or PDF for short.
"""

# %%
mp_vectors = [samps.median_pdf_vector for samps in agg_filter.values("samples")]
mp_instances = [samps.median_pdf_instance for samps in agg_filter.values("samples")]

print("Most Probable Model Parameter Lists:\n")
print(mp_vectors, "\n")
print("Most probable Model Instances:\n")
print(mp_instances, "\n")

# %%
"""
We can compute the upper and lower errors on each parameter at a given sigma limit. These are computed via 
marginalization, whereby instead of using the median of the histogram (e.g. the parameter value at the 50% mark of the
histogram) the values at a specified sigma interval are used. For 3 sigma, these confidence intervals are at 0.3% and
99.7%.

# Here, I use "ue3" to signify this is an upper error at 3 sigma confidence,, and "le3" for the lower error.
"""

# %%
ue3_vectors = [
    out.error_vector_at_upper_sigma(sigma=3.0) for out in agg_filter.values("samples")
]
ue3_instances = [
    out.error_instance_at_upper_sigma(sigma=3.0) for out in agg_filter.values("samples")
]
le3_vectors = [
    out.error_vector_at_lower_sigma(sigma=3.0) for out in agg_filter.values("samples")
]
le3_instances = [
    out.error_instance_at_lower_sigma(sigma=3.0) for out in agg_filter.values("samples")
]

print("Errors Lists:\n")
print(ue3_vectors, "\n")
print(le3_vectors, "\n")
print("Errors Instances:\n")
print(ue3_instances, "\n")
print(le3_instances, "\n")

# %%
"""
Lets end part 1 with something more ambitious. Lets create a plot of the inferred sigma values vs intensity of each
Gaussian profile, including error bars at 3 sigma confidence.
"""

# %%
import matplotlib.pyplot as plt

mp_instances = [samps.median_pdf_instance for samps in agg_filter.values("samples")]
ue3_instances = [
    out.error_instance_at_upper_sigma(sigma=3.0) for out in agg_filter.values("samples")
]
le3_instances = [
    out.error_instance_at_lower_sigma(sigma=3.0) for out in agg_filter.values("samples")
]

mp_sigmas = [instance.gaussian.sigma for instance in mp_instances]
ue3_sigmas = [instance.gaussian.sigma for instance in ue3_instances]
le3_sigmas = [instance.gaussian.sigma for instance in le3_instances]
mp_intensitys = [instance.gaussian.sigma for instance in mp_instances]
ue3_intensitys = [instance.gaussian.sigma for instance in ue3_instances]
le3_intensitys = [instance.gaussian.intensity for instance in le3_instances]

plt.errorbar(
    x=mp_sigmas,
    y=mp_intensitys,
    marker=".",
    linestyle="",
    xerr=[le3_sigmas, ue3_sigmas],
    yerr=[le3_intensitys, ue3_intensitys],
)
plt.show()
