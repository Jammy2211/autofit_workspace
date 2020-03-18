import autofit as af
import numpy as np

from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.model import (
    gaussian,
)

from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.phase import (
    phase as ph,
)

# In the previous tutorial, we used PyAutoFit to fit a 1D Gaussian model to a dataset. In this tutorial, we'll repeat
# the same fit, but extend the phase module to perform a number of additional tasks:

# - Masking: The phase is passed a mask such that regions of the dataset are omitted.
# - Visualization: Images showing the model fit are output on-the-fly during the non-linear search.

# These new features have lead to an additional modules in the 'phase' package not present in tutorial 3, called
# 'visualizer.py'. Before looking at this module, lets perform a fit to see the changed behaviour of PyAutoFit.

# You need to change the path below to the chapter 1 directory so we can load the dataset
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

dataset_path = chapter_path + "dataset/gaussian_x1/"

from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

# Before fitting data, we may want to mask it, removing regions of the data we know are defective or where there is no
# signal.

# To facilitate this we have added the module 'masked_dataset.py' to our dataset package. This takes our dataset
# and a mask and combines the two to create a masked dataset. The fit.py module has also been updated to use a mask
# during the fit. Check them both out now to see how the mask is used!

# Masking occurs within the phase package of PyAutoFit, which we'll inspect at the end of the tutorial. However,
# for a phase to run it now requires that a mask is passed to it. For this tutorial, lets create a which removes the
# last 30 data-points in our data.

# (In our convention, a mask value of 'True' means it IS masked and thus removed from the fit).

mask = np.full(fill_value=False, shape=dataset.data.shape)
mask[-30:] = True

# Lets now reperform the fit from tutorial 3, but with a masked dataset and visualization.

phase = ph.Phase(phase_name="phase_t4", gaussian=af.PriorModel(gaussian.Gaussian))

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t4"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)


# Note that we are passing our mask to the phase run function, which we did not in previous tutorials.

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# Lets check that this phase did indeed perform visualization. Navigate to the folder 'image' in the directory
# above. You should now see a set of folders containing visualization of the dataset and fit. As promised, our phase is
# now taking care of the visualization of our model.

# Visualization happens 'on-the-fly', such that during MultiNest these images are output using the current best-fit
# model MultiNest has found. For models more complex than our 2D Gaussian this is useful, as it means we can check
# that MultiNest has found reasonable solutions during a run and can thus cancel it early if it has ended up with an
# incorrect solution.

# How often does PyAutoFit output new images? This is set by the 'visualize_interval' in the config file
# 'config/visualize/general.ini'

# Finally, now inspect the 'phase.py', 'analysis.py' and 'visualizer.py' modules in the source code. Here, it should
# become how the masked data is set up and how visualization is performed.

# And with that, we have completed this (fairly short) tutorial. There are two things worth ending on:

# 1) In tutorial 2 onwards, we introduced the 'plot' package that had functioons specific to plotting attributes of
#    a data-set and fit. We argued this would help us in later tutorials, and this is the first tutorial that it has,
#    as it made it straight forward to perform plotting with the Visualizer. For your model-fitting project you should
#    aim to strichtly adhere to performing all plots in a 'plot' module - this will benefit even more in tutorial 6.

# 2) For our very simple 1D case, we used a 1D NumPy array to represent a mask. For projects with more complicated
#    datasets, one may require more complicated masks, warranting a 'mask' package and 'mask.py' module. In tutorial 8
#    we will show an example of this.
