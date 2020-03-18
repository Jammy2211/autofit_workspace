import autofit as af
import numpy as np

from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.model import (
    profiles,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.dataset import (
    dataset as ds,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.phase import (
    phase as ph,
)

# In this tutorial, we're going to add input parameters to a Phase object that customizes the analysis. We'll use the
# specific example of two input parameters that trim our data-set from the left and right before fitting it. This
# example is somewhat trivial (afterall, we could achieve almost the same effect with masking), but it will serve to
# illustrate phase customization.

# When we customize a phase, we'll use these input parameters to perform phase tagging. Here, we 'tag' the output
# path of the phase's results, such that every time a phase is run with a different customization a new set of
# unique results are stored for those settings. For a given data-set we are thus able to fit it multiple times using
# different settings to compare the results.

# These new features have lead to additional modules in the 'phase' package called 'meta_dataset.py' and 'tagging.py'.
# Before looking at these modules, lets first perform a series of MultiNest fits to see how they change the behaviour
# of PyAutoFit.

# You need to change the path below to the chapter 1 directory so we can load the dataset
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

# We're now going to perform multiple fits, where each fit trims the data-set that is fitted.
# To do this, we'll set up phases with the phase-settings 'data_trim_left' and 'data_trim_right'.

# - data_trim_left:

#       The dataset's image and noise-map are trimmed and removed from the left (e.g. 1d index values from 0).
#       For example, if the dataset has shape (100,) and we set data_trim_left=10, the dataset that is fitted will have
#       shape (90,). The mask is trimmed in the same way.

# - data_trim_right:

#       This behaves the same as data_trim_left, however data is removed from the right (e.g. 1D index values from the
#       shape of the 1D data).

# For our first phase, we will omit both the phase setting (by setting it to None) and perform the fit from tutorial
# 3 where we fit a single Gaussian profile to data composed of a single Gaussian (unlike tutorial 3, we'll use a
# CollectionPriorModel to do this).

phase = ph.Phase(
    phase_name="phase_t6",
    profiles=af.CollectionPriorModel(gaussian=profiles.Gaussian),
    data_trim_left=None,
    data_trim_right=None,
)

# Lets load the dataset, create a mask and perform the fit.

dataset_path = chapter_path + "dataset/gaussian_x1/"

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

mask = np.full(fill_value=False, shape=dataset.data.shape)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# Okay, lets look at what happened differently in this phase. To begin, lets note the output directory:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_6_phase_customization/output/phase_t6/phase_tag'

# There is a small change to this directory compared to tutorial 5, there is a new folder 'phase_tag' within which the
# results are stored. It'll be clear why this is in a moment.

# Next, we're going to customize and run a phase using the data_trim_left and right parameters. We create a new phase
# using these parameters and run it.

phase = ph.Phase(
    phase_name="phase_t6",
    profiles=af.CollectionPriorModel(gaussian=profiles.Gaussian),
    data_trim_left=20,
    data_trim_right=30,
)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# You'll note the results are now in a slightly different directory to the fit performed above:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_6_phase_customization/output/phase_example/phase_tag__trim_left_20__trim_right_30'

# By customizing our phase's settings, PyAutoFit has changed it output path using a tag for this phase. There are two
# reasons PyAutoFit does this:

# 1) Tags describes the analysis, making it explicit what was done to the dataset for the fit.

# 2) Tags create a unique output path, allowing you to compare results of phases that use different settings. Equally,
#    if you run multiple phases with different settings this ensures the non-linear search (e.g. MultiNest) won't
#    inadvertantly use results generated via a different analysis method.

# In this tutorial, the phase setting changed the data-set that was fitted. However, phase settings do not necessarily
# need to customize the data-set. For example, they could control some aspect of the model, for example the precision
# by which the model image is numerically calculated. For more complex fitting procedures, settings could control
# whether certain features are used, which when turned on / off reduce the accuracy of the model at the expensive of
# greater computational run-time.

# Phase settings are project specific and it could well be your modeling problem is simple enough not to need them.
# However, if it does, remember that phase settings are a powerful means to fit models using different settings and
# compare whether a setting does or does not change the model inferred. In later chapters, we'll discuss more complex
# model-fitting procedures that could use 'fast' less accurate settings to initialize the model-fit, but switch to
# slower more accurate settings later on.
