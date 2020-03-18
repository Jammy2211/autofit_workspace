import autofit as af

from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.model import profiles
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.dataset import (
    dataset as ds,
)
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.phase import phase as ph

import numpy as np

# After fitting a large suite of data with the same pipeline, the aggregator allows us to load the results and
# manipulate / plot them using a Python script or Jupyter notebook.

# To begin, we need a set of results that we want to analyse using the aggregator. We'll create this by fitting 3
# different data-sets. Each dataset is a single Gaussian and we'll fit them using a single Gaussian model.

# You need to change the path below to the chapter 1 directory so we can load the dataset
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

# Here, for each dataset we are going to set up the correct path, load it, create its mask and fit it with the phase above.

# We want our results to be in a folder specific to the dataset. We'll use the dataset_name string to do this. Lets
# create a list of all 3 of our dataset names.
dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

# This for loop runs over every dataset name, checkout the comments below for how we set up the path structure.

for dataset_name in dataset_names:

    # The code below loads the dataset and creates the mask as per usual.
    dataset_path = chapter_path + "dataset/" + dataset_name

    dataset = ds.Dataset.from_fits(
        data_path=dataset_path + "/data.fits",
        noise_map_path=dataset_path + "/noise_map.fits",
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

# phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# Checkout the output folder - you should see three new sets of results correspoonding to our 3 Gaussian datasets.
# Unlike previous tutorials, these folders in the output folder are named after the dataset and contain the folder
# with the phase'sname, as opposed to just the phase-name folder.

# To load these results with the aggregator, we simply point it to the path of the results we want it to inspect.

output_path = chapter_path + "/output/"

aggregator = af.Aggregator(directory=str(output_path))

# We can now create a list of the 'non-linear outputs' of every fit. An instance of the NonLinearOutput class acts as
# an interface between the results of the non-linear fit on your hard-disk and Python.

# The fits to each lens used MultiNest, so below we create a list of instances of the MultiNestOutput class (the
# non-linear output class will change dependent on the non-linear optimizer used).
outputs = aggregator.output

# When we print this list of outputs you should see over 3 different MultiNestOutput instances. There are more than 3
# because the aggregator has loaded the results of previous tutorial as well as the 3 fits we performed abovde!
print("MultiNest Outputs:\n")
print(outputs)
print("Total Outputs = ", len(outputs), "\n")

# To remove the fits of previous tutorials and just keep the MultiNestOutputs of the 3 datasets fitted in this tutorial
# We need to us the aggregator's filter tool. The phase name 'phase_t7' used in this tutorial is unique to all 3 fits,
# so we can use it to filter our results are desired.
phase_name = "phase_t7"
outputs = aggregator.filter(phase=phase_name).output

# As expected, this list now has only 3 MultiNestOutputs, one for each dataset we fitted.
print("Phase Name Filtered MultiNest Outputs:\n")
print(outputs)
print("Total Outputs = ", len(outputs), "\n")

# We can use the outputs to create a list of the most-likely (e.g. highest likelihood) model of each fit to our three
# images (in this case in phase 3).
vector = [out.most_likely_vector for out in outputs]
print("Most Likely Model Parameter Lists:\n")
print(vector, "\n")

# This provides us with lists of all model parameters. However, this isn't that much use - which values correspond
# to which parameters?

# Its more useful to create the model instance of every fit.
instances = [out.most_likely_instance for out in outputs]
print("Most Likely Model Instances:\n")
print(instances, "\n")

# A model instance contains all the model components of our fit - which for the fits above was a single gaussian
# profile (the word 'gaussian' comes from what we called it in the CollectionPriorModel when making the phase above).
print(instances[0].profiles.gaussian)
print(instances[1].profiles.gaussian)
print(instances[2].profiles.gaussian)

# This, of course, gives us access to any individual parameter of our most-likely model. Below, we see that the 3
# Gaussians were simulated using sigma values of 1.0, 5.0 and 10.0.
print(instances[0].profiles.gaussian.sigma)
print(instances[1].profiles.gaussian.sigma)
print(instances[2].profiles.gaussian.sigma)

# We can also access the 'most probable' model, which is the model computed by marginalizing over the MultiNest samples
# of every parameter in 1D and taking the median of this PDF.
mp_vectors = [out.most_probable_vector for out in outputs]
mp_instances = [out.most_probable_instance for out in outputs]

print("Most Probable Model Parameter Lists:\n")
print(mp_vectors, "\n")
print("Most probable Model Instances:\n")
print(mp_instances, "\n")

# We can compute the upper and lower errors on each parameter at a given sigma limit.
ue_vectors = [out.error_vector_at_upper_sigma(sigma=3.0) for out in outputs]
ue_instances = [out.error_instance_at_upper_sigma(sigma=3.0) for out in outputs]
le_vectors = [out.error_vector_at_lower_sigma(sigma=3.0) for out in outputs]
le_instances = [out.error_instance_at_lower_sigma(sigma=3.0) for out in outputs]

print("Errors Lists:\n")
print(ue_vectors, "\n")
print(le_vectors, "\n")
print("Errors Instances:\n")
print(ue_instances, "\n")
print(le_instances, "\n")

# The maximum likelihood of each model fit and its Bayesian evidence (estimated via MultiNest) are also available.

# Given each fit is to a different image, these are not very useful. However, in tutorial 5 we'll look at using the
# aggregator for images that we fit with many different models and many different pipelines, in which case comparing
# the evidences allows us to perform Bayesian model comparison!
print("Likelihoods:\n")
print([out.maximum_log_likelihood for out in outputs])
print([out.evidence for out in outputs])

# We can also print the "model_results" of all phases, which is string that summarizes every fit's lens model providing
# quick inspection of all results.
results = aggregator.filter(phase=phase_name).model_results
print("Model Results Summary:\n")
print(results, "\n")

# Lets end part 1 with something more ambitious. Lets create a plot of the inferred sigma values vs intensity of each
# Gaussian profile, including error bars at 3 sigma confidence.

import matplotlib.pyplot as plt

mp_instances = [out.most_probable_instance for out in outputs]
ue_instances = [out.error_instance_at_upper_sigma(sigma=3.0) for out in outputs]
le_instances = [out.error_instance_at_lower_sigma(sigma=3.0) for out in outputs]

einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in mp_instances
]
einstein_radii_upper = [
    instance.galaxies.lens.mass.einstein_radius for instance in ue_instances
]
einstein_radii_lower = [
    instance.galaxies.lens.mass.einstein_radius for instance in le_instances
]
axis_ratios = [instance.galaxies.lens.mass.axis_ratio for instance in mp_instances]
axis_ratios_upper = [
    instance.galaxies.lens.mass.axis_ratio for instance in ue_instances
]
axis_ratios_lower = [
    instance.galaxies.lens.mass.axis_ratio for instance in le_instances
]

plt.errorbar(
    x=einstein_radii, y=axis_ratios, xerr=einstein_radii_upper, yerr=axis_ratios_upper
)
