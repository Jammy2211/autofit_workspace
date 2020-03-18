import autofit as af

from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.dataset import (
    masked_dataset as md,
)
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.fit import fit as f
from howtofit.chapter_1_introduction.tutorial_7_aggregator.src.plot import (
    dataset_plots,
    fit_plots,
)

import numpy as np

# In part 1 of tutorial 7, we fitted 3 datasets and used the aggregator to load their results. We focused on the
# results of the non-linear search, MultiNest. In part 2, we'll look at how the way we designed our source code
# makes it easy to use these results to plot results and data.

# You need to change the path below to the chapter 1 directory so we can load the dataset.
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

# To load these results with the aggregator, we again point it to the path of the results we want it to inspect.

output_path = chapter_path + "/output/"

aggregator = af.Aggregator(directory=str(output_path))
phase_name = "phase_t7"

# We can use the aggregator to load every fit's dataset, by changing the 'output' attribute to the 'dataset' attribute
# at the end of the aggregator. We'll filter by phase name again to get datasets of only the fits performed for this
# tutorial.
datasets = aggregator.filter(phase=phase_name).dataset
print("Datasets:")
print(datasets, "\n")

# It is here the object-based design of our plot module comes into its own. We have the data-set objects loaded, meaning
# we can easily plot each dataset using the 'dataset_plot.py' module.
[dataset_plots.data(dataset=dataset) for dataset in datasets]

# We can repeat the same trick to get the mask of every fit.
masks = aggregator.filter(phase=phase_name).mask
print("Masks:")
print(masks, "\n")

# We're going to refir our datasets using the best-fit model of each phase. To do this, we'll need each phase's masked
# dataset.

# (If you are unsure what the 'zip' is doing below, it essentially combines the'datasets' and 'masks' lists in such
# a way that we can iterate over the two simultaneously to create each MaskedDataset).
masked_datasets = [
    md.MaskedDataset(dataset=dataset, mask=mask)
    for dataset, mask in zip(datasets, masks)
]

# Lets filter by phase name again to get the the most-likely model instances, as we did in part 1.
outputs = aggregator.filter(phase=phase_name).output
instances = [out.most_likely_instance for out in outputs]

# Okay, we want to inspect the fit of each best-fit model. To do this, we reperform each fit.

# First, we need to create the model-data of every best-fit model instance. Lets begin by creating a list of profiles of
# every phase.
profiles = [instance.profiles for instance in instances]

# We can use these to create the model data of each set of profiles (Which in this case is just 1 Gaussian, but had
# we included more profiles in the model would consist of multiple Gaussians / Exponentials).
model_datas = [
    profile.line_from_xvalues(xvalues=dataset.xvalues)
    for profile, dataset in zip(profiles, datasets)
]

# And, as we did in tutorial 2, we can combine the masked_datasets and model_datas in a Fit object to create the
# best-fit fit of each phase!
fits = [
    f.DatasetFit(masked_dataset=masked_dataset, model_data=model_data)
    for masked_dataset, model_data in zip(masked_datasets, model_datas)
]

# We can now plot different components of the fit (again benefiting from how we set up the 'fit_plots.py' module)!
[fit_plots.residual_map(fit=fit) for fit in fits]
[fit_plots.normalized_residual_map(fit=fit) for fit in fits]
[fit_plots.chi_squared_map(fit=fit) for fit in fits]

# Setting up the above objects (the masked_datasets, profiles, model datas, fits) was a bit of work. It wasn't too many
# lines of code, but for something we'll likely want to do many times it'd be nice to have a short cut to setting
# them up, right?

# In 'aggregator.py' we've set up exactly such a short=cut. This essentially reperforms the commands above to set up
# each list, give the aggregator. You'll note that we had actually already one this for loading the dataset and mask via
# the aggregator above.
#
# This means we can setup the above objects using the following single lines of code:

masked_datasets = aggregator.filter(phase=phase_name).masked_dataset
profiles = aggregator.filter(phase=phase_name).profile
model_datas = aggregator.filter(phase=phase_name).model_data
fits = aggregator.filter(phase=phase_name).fit

# For your model-fitting project, you'll need to update the aggregator in the same way. Again, to emphasise this point,
# this is why we have emphasised the object-oriented design of our model-fitting project through. This design makes it
# very easy to inspect results via the aggregator later on!
