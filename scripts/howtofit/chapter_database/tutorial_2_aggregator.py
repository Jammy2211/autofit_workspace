"""
Tutorial 2: Aggregator
======================

In the previous tutorial, we fitted 3 datasets with an identical `NonLinearSearch`, outputting the results of each to a
unique folder on our hard disk.

In this tutorial, we'll use the `Aggregator` to load the `Result`'s and manipulate them using our Jupyter
notebook. The API for using a `Result` is described fully in tutorial 6 of chapter 1 of **HowToFit**.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Gaussian` and `plot_profile_1d` objects and functions you 
have seen and used elsewhere throughout the workspace.
"""

"""
__Building a Database File From an Output Folder__

In the previous tutorials, we built the database file `chapter_database.sqlite` via the results output to
hard-disk.

We can therefore simply load this database from the hard-disk in order to use the aggregator.
"""
database_name = "chapter_database"

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

"""
__Generators__

Before using the aggregator to inspect results, let me quickly cover Python generators. A generator is an object that 
iterates over a function when it is called. The aggregator creates all of the objects that it loads from the database 
as generators (as opposed to a list, or dictionary, or other Python type).

Why? Because lists and dictionaries store every entry in memory simultaneously. If you fit many datasets, this will use 
a lot of memory and crash your laptop! On the other hand, a generator only stores the object in memory when it is used; 
Python is then free to overwrite it afterwards. Thus, your laptop won't crash!

There are two things to bare in mind with generators:

1) A generator has no length and to determine how many entries it contains you first must turn it into a list.

2) Once we use a generator, we cannot use it again and need to remake it. For this reason, we typically avoid 
 storing the generator as a variable and instead use the aggregator to create them on use.

We can now create a `samples` generator of every fit. The `results` example scripts show how , an instance of 
the `Samples` class acts as an interface to the results of the non-linear search.
"""
samples_gen = agg.values("samples")

"""
When we print this list of outputs you should see over 3 different `SamplesNest` instances, corresponding to the 3
model-fits we performed in the previous tutorial.
"""
print("Emcee Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(agg), "\n")

"""
__Samples__

We've encountered the `Samples` class in previous tutorials. As we saw in chapter 1, the `Samples` class contains all 
the accepted parameter samples of the `NonLinearSearch`, which is a list of lists where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.

With the `Aggregator` we can now get information on the `Samples` of all 3 model-fits, as opposed to just 1 fit using 
its `Result` object.
"""
for samples in agg.values("samples"):
    print("All parameters of the very first sample")
    print(samples.parameter_lists[0])
    print("The tenth sample`s third parameter")
    print(samples.parameter_lists[9][2])
    print()

"""
We can use the `Aggregator` to get information on the `log_likelihood_list`, log_prior_list`, `weight_list`, etc. of 
every fit.
"""
for samples in agg.values("samples"):
    print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
    print(samples.log_likelihood_list[9])
    print(samples.log_prior_list[9])
    print(samples.log_posterior_list[9])
    print(samples.weight_list[9])
    print()

"""
__Instances__

We can use the `Aggregator` to create a list of the `max_log_likelihood` instances of every fit.
"""
max_lh_instance_list = [samps.max_log_likelihood() for samps in agg.values("samples")]

print("Maximum Log Likelihood Model Instances:\n")
print(max_lh_instance_list, "\n")

"""
The model instance contains all the model components of our fit which for the fits above was a single `Gaussian`
profile (the word `gaussian` comes from what we called it in the `Collection` above).
"""
print(max_lh_instance_list[0].gaussian)
print(max_lh_instance_list[1].gaussian)
print(max_lh_instance_list[2].gaussian)

"""
This, of course, gives us access to any individual parameter of our maximum log likelihood `instance`. Below, we see 
that the 3 `Gaussian`s were simulated using `sigma` values of 1.0, 5.0 and 10.0.
"""
print(max_lh_instance_list[0].gaussian.sigma)
print(max_lh_instance_list[1].gaussian.sigma)
print(max_lh_instance_list[2].gaussian.sigma)

"""
__Vectors__

We can use the outputs to create a list of the maximum log likelihood model of each fit to our three images.
"""
vector_list = [
    samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
]
print("Maximum Log Likelihood Parameter Lists:\n")
print(vector_list, "\n")


"""
__Median PDF__

We can also access the `median_pdf` model via the `Aggregator`, as we saw for the `Samples` object in chapter 1.
"""
median_pdf_instance_list = [samps.median_pdf() for samps in agg.values("samples")]

print("Median PDF Model Instances:\n")
print(median_pdf_instance_list, "\n")

"""
__Ordering__

The default ordering of the results can be a bit random, as it depends on how the sqlite database is built. 

The `order_by` method can be used to order by a property of the database that is a string, for example by ordering 
using the `unique_tag` (which we set up in the search as the `dataset_name`) the database orders results alphabetically
according to dataset name.
"""
agg = agg.order_by(agg.search.unique_tag)

"""
We can also order by a bool, for example making it so all completed results are at the front of the aggregator.
"""
agg = agg.order_by(agg.search.is_complete)

"""
__Errors__

Lets try something more ambitious and create a plot of the inferred `sigma` values vs `normalization` of each `Gaussian` 
profile, including error bars at $3\sigma$ confidence.
"""
import matplotlib.pyplot as plt

median_pdf_instance_list = [samps.median_pdf() for samps in agg.values("samples")]
ue3_instance_list = [
    samp.errors_at_upper_sigma(sigma=3.0) for samp in agg.values("samples")
]
le3_instance_list = [
    samp.errors_at_lower_sigma(sigma=3.0) for samp in agg.values("samples")
]

mp_sigma_list = [instance.gaussian.sigma for instance in median_pdf_instance_list]
ue3_sigma_list = [instance.gaussian.sigma for instance in ue3_instance_list]
le3_sigma_list = [instance.gaussian.sigma for instance in le3_instance_list]
mp_normalization_list = [
    instance.gaussian.normalization for instance in median_pdf_instance_list
]
ue3_normalization_list = [
    instance.gaussian.normalization for instance in ue3_instance_list
]
le3_normalization_list = [
    instance.gaussian.normalization for instance in le3_instance_list
]

plt.errorbar(
    x=mp_sigma_list,
    y=mp_normalization_list,
    marker=".",
    linestyle="",
    xerr=[le3_sigma_list, ue3_sigma_list],
    yerr=[le3_normalization_list, ue3_normalization_list],
)
plt.show()

"""
__Samples Filtering__

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = list(agg.values("samples"))[0]

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

This is how the source code internally stores the path to different components of the model, but it is not in-line 
with the PyAutoFIT API used to compose a model.

We can alternatively use the following API:
"""
samples = list(agg.values("samples"))[0]

samples = samples.with_paths(["gaussian.centre"])

print("All parameters of the very first sample (containing only the Gaussian centre.")
print(samples.parameter_lists[0])

"""
Above, we filtered the `Samples` but asking for all parameters which included the path ("gaussian", "centre").

We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the Gaussian's `centre` to be left with 2 parameters; the `normalization` and `sigma`.
"""
samples = list(agg.values("samples"))[0]

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
We can keep and remove entire paths of the samples, for example keeping only the parameters of the `Gaussian`.

For this example this is somewhat trivial, given the model only contains the `Gaussian`, but for models with
multiply components (e.g. a `Collection`) and multi-level models this can be a powerful way to extract samples.
"""
samples = list(agg.values("samples"))[0]
samples = samples.with_paths([("gaussian",)])
print("Parameters of the first sample of the Gaussian model component")
print(samples.parameter_lists[0])

"""
___Wrap Up__

With that, tutorial 2 is complete. 

The take home point of this tutorial is that everything that is available in a `Result` or `Samples` object is 
accessible via the `Aggregator`. 
"""
