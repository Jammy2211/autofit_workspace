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
import autofit as af

from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.model import profiles
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.dataset import (
    dataset as ds,
)
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.phase import phase as ph

import numpy as np

# %%
"""
You need to change the path below to the chapter 1 directory so we can load the dataset
"""

# %%
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

# %%
"""
Here, for each dataset we are going to set up the correct path, load it, create its mask and fit it with the phase above.

We want our results to be in a folder specific to the dataset. We'll use the dataset_name string to do this. Lets
create a list of all 3 of our dataset names.

We'll also pass these names to the dataset when we create it - the name will be used by the aggregator to name the file
the data is stored. More importantly, the name will be accessible to the aggregator, and we will use it to label 
figures we make via the aggregator.
"""

# %%
dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

# %%
"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the dataset) that isn't part of the model-fit is made accessible to the 
aggregator. For example, below we write info on the dataset's data of observation and exposure time.
"""
info = {"date_of_observation" : "01-02-18", "exposure_time" : 1000.0}

# %%
"""
This for loop runs over every dataset name, checkout the comments below for how we set up the path structure.
"""

# %%
for dataset_name in dataset_names:

    # The code below loads the dataset and creates the mask as per usual.

    dataset_path = chapter_path + "dataset/" + dataset_name

    dataset = ds.Dataset.from_fits(
        data_path=dataset_path + "/data.fits",
        noise_map_path=dataset_path + "/noise_map.fits",
        name=dataset_name,
    )

    mask = np.full(fill_value=False, shape=dataset.data.shape)

    # Here, we create a phase as we are used to. However, we also include an input parameter 'phase_folders'. The
    # phase folders define the names of folders that the phase goes in. For example, if a phase goes to the path:

    # '/path/to/autofit_workspace/output/phase_name/'

    # A phase folder with the input 'phase_folder' edits this path to:

    # '/path/to/autofit_workspace/output/phase_folder/phase_name/'

    # You can input multiple phase folders, for example 'phase_folders=['folder_0', 'folder_1'] would create the path:

    # '/path/to/autofit_workspace/output/folder_0/folder_1/phase_name/'

    # Below, we use the data_name, so as required our results go in a folder specific to the dataset, e.g:

    # '/path/to/autofit_workspace/output/gaussian_x1_0/phase_t7/'

    print(
        "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/"
        + dataset_name
        + "/phase_t7"
        "folder for live output of the results."
        "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
    )

    phase = ph.Phase(
        phase_name="phase_t7",
        phase_folders=[dataset_name],
        profiles=af.CollectionPriorModel(gaussian=profiles.Gaussian),
    )

    # Note that we pass the info to the phase when we run it, so that the aggregator can make it accessible.

    phase.run(dataset=dataset, mask=mask, info=info)

print("MultiNest has finished run - you may now continue the notebook.")

# %%
"""
Checkout the output folder - you should see three new sets of results correspoonding to our 3 Gaussian datasets.
Unlike previous tutorials, these folders in the output folder are named after the dataset and contain the folder
with the phase'sname, as opposed to just the phase-name folder.

To load these results with the aggregator, we simply point it to the path of the results we want it to inspect.
"""

# %%
output_path = chapter_path + "/output/"

agg = af.Aggregator(directory=str(output_path))

# %%
"""
To begin, let me quickly explain what a generator is in Python, for those unaware. A generator is an object that 
iterates over a function when it is called. The aggregator creates all objects as generators, rather than lists, or 
dictionaries, or whatever.

Why? Because lists store every entry in memory simultaneously. If you fit many datasets, you'll have lots of results and 
therefore use a lot of memory. This will crash your laptop! On the other hand, a generator only stores the object in 
memory when it runs the function; it is free to overwrite it afterwards. This, your laptop won't crash!

There are two things to bare in mind with generators:

1) A generator has no length, thus to determine how many entries of data it corresponds to you first must turn it to a 
list.

2) Once we use a generator, we cannot use it again - we'll need to remake it.

We can now create a 'non-linear outputs' generator of every fit. An instance of the NonLinearOutput class acts as an 
interface between the results of the non-linear fit on your hard-disk and Python.
"""

# %%
output_gen = agg.values("output")

# %%
"""
When we print this list of outputs you should see over 3 different MultiNestOutput instances. There are more than 3
because the aggregator has loaded the results of previous tutorial as well as the 3 fits we performed abode!
"""

# %%
print("MultiNest Outputs:\n")
print(output_gen)
print("Total Outputs = ", len(list(output_gen)), "\n")

# %%
"""
To remove the fits of previous tutorials and just keep the MultiNestOutputs of the 3 datasets fitted in this tutorial
We need to us the aggregator's filter tool. The phase name 'phase_t7' used in this tutorial is unique to all 3 fits,
so we can use it to filter our results are desired.
"""


# %%
phase_name = "phase_t7"
agg_filter = agg.filter(agg.phase == phase_name)
output_gen = agg_filter.values("output")

# %%
"""
As expected, this list now has only 3 MultiNestOutputs, one for each dataset we fitted.
"""

# %%
print("Phase Name Filtered MultiNest Outputs:\n")
print(output_gen)
print("Total Outputs = ", len(list(output_gen)), "\n")

# %%
"""
We can use the outputs to create a list of the most-likely (e.g. highest likelihood) model of each fit to our three
images (in this case in phase 3).
"""

# %%
vector = [out.most_likely_vector for out in agg_filter.values("output")]
print("Most Likely Model Parameter Lists:\n")
print(vector, "\n")

# %%
"""
This provides us with lists of all model parameters. However, this isn't that much use - which values correspond
to which parameters?

Its more useful to create the model instance of every fit.
"""

# %%
instances = [out.most_likely_instance for out in agg_filter.values("output")]
print("Most Likely Model Instances:\n")
print(instances, "\n")

# %%
"""
A model instance contains all the model components of our fit - which for the fits above was a single gaussian
profile (the word 'gaussian' comes from what we called it in the CollectionPriorModel when making the phase above).
"""

# %%
print(instances[0].profiles.gaussian)
print(instances[1].profiles.gaussian)
print(instances[2].profiles.gaussian)

# %%
"""
This, of course, gives us access to any individual parameter of our most-likely model. Below, we see that the 3
Gaussians were simulated using sigma values of 1.0, 5.0 and 10.0.
"""

# %%
print(instances[0].profiles.gaussian.sigma)
print(instances[1].profiles.gaussian.sigma)
print(instances[2].profiles.gaussian.sigma)

# %%
"""
We can also access the 'most probable' model, which is the model computed by marginalizing over the MultiNest samples
of every parameter in 1D and taking the median of this PDF.
"""

# %%
mp_vectors = [out.most_probable_vector for out in agg_filter.values("output")]
mp_instances = [out.most_probable_instance for out in agg_filter.values("output")]

print("Most Probable Model Parameter Lists:\n")
print(mp_vectors, "\n")
print("Most probable Model Instances:\n")
print(mp_instances, "\n")

# %%
"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

# Here, I use "ue3" to signify this is an upper error at 3 sigma confidence,, and "le3" for the lower error.
"""

# %%
ue3_vectors = [out.error_vector_at_upper_sigma(sigma=3.0) for out in agg_filter.values("output")]
ue3_instances = [out.error_instance_at_upper_sigma(sigma=3.0) for out in agg_filter.values("output")]
le3_vectors = [out.error_vector_at_lower_sigma(sigma=3.0) for out in agg_filter.values("output")]
le3_instances = [out.error_instance_at_lower_sigma(sigma=3.0) for out in agg_filter.values("output")]

print("Errors Lists:\n")
print(ue3_vectors, "\n")
print(le3_vectors, "\n")
print("Errors Instances:\n")
print(ue3_instances, "\n")
print(le3_instances, "\n")

# %%
"""
The maximum likelihood of each model fit and its Bayesian evidence (estimated via MultiNest) are also available.

Given each fit is to a different image, these are not very useful. However, in tutorial 5 we'll look at using the
aggregator for images that we fit with many different models and many different pipelines, in which case comparing
the evidences allows us to perform Bayesian model comparison!
"""

# %%
print("Likelihoods:\n")
print([out.maximum_log_likelihood for out in agg_filter.values("output")])
print([out.evidence for out in output_gen])

# %%
"""
We can also print the "model_results" of all phases, which is string that summarizes every fit's lens model providing
quick inspection of all results.
"""

# %%
results = agg_filter.model_results
print("Model Results Summary:\n")
print(results, "\n")

# %%
"""
Lets end part 1 with something more ambitious. Lets create a plot of the inferred sigma values vs intensity of each
Gaussian profile, including error bars at 3 sigma confidence.
"""

# %%
import matplotlib.pyplot as plt

mp_instances = [out.most_probable_instance for out in agg_filter.values("output")]
ue3_instances = [out.error_instance_at_upper_sigma(sigma=3.0) for out in agg_filter.values("output")]
le3_instances = [out.error_instance_at_lower_sigma(sigma=3.0) for out in agg_filter.values("output")]

mp_sigmas = [
    instance.profiles.gaussian.sigma for instance in mp_instances
]
ue3_sigmas = [
    instance.profiles.gaussian.sigma for instance in ue3_instances
]
le3_sigmas = [
    instance.profiles.gaussian.sigma for instance in le3_instances
]
mp_intensitys = [instance.profiles.gaussian.sigma for instance in mp_instances]
ue3_intensitys = [
    instance.profiles.gaussian.sigma for instance in ue3_instances
]
le3_intensitys = [
    instance.profiles.gaussian.intensity for instance in le3_instances
]

plt.errorbar(
    x=mp_sigmas, y=mp_intensitys, marker=".", linestyle="",
    xerr=[le3_sigmas, ue3_sigmas], yerr=[le3_intensitys, ue3_intensitys]
)
plt.show()
