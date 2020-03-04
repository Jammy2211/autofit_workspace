import autofit as af
import autoarray as aa

from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.model import (
    gaussians,
)

from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase import (
    phase as ph,
)

# In the previous tutorial, we used PyAutoFit to fit a 2D Gaussian model to our data. In this tutorial, we'll repeat
# the same fit, but extend phases to perform a number of additional tasks:

# - Masking: The phase is passed a mask such that regions of the image are not fitted.
# - Visualization: Images showing the model fit are output on-the-fly during the non-linear search.
# - Customization: The fit is customized to alter the dataset before fitting.
# - Tagging: The output paths of the phase are tagged depending on the fit customization.

# These new features have lead to additional modules in the 'phase' package not present in tutorial 3, called
# 'meta_dataset.py', 'tagging.py' and 'visualizer.py'. Before looking at these modules, lets first perform a series of
# MultiNest fits to see how they change the behaviour of PyAutoFit.

# You need to change the path below to the chapter 1 directory so we can load the dataset
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config", output_path=chapter_path + "output"
)

dataset_path = chapter_path + "dataset/gaussian_x1/"

from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    pixel_scales=1.0,
)

# Before fitting data, we often want mask it, removing regions of the data where we know it is defective or where
# there is no signal to save on computational run time.

# To facilitate this we have added the module 'masked_dataset.py' to our dataset package. This takes our dataset
# dataset and a mask, and combines the two to create a masked dataset. Check it out now!

# This masking occurs within the phase package of PyAutoFit, which we'll inspect at the end of the tutorial. However,
# for a phase to run it does now require that a mask is passed to it. For this tutorial, lets use a mask of size 10.0
# (in Gaussian units set by the pixel_scales of the dataset as specified above).

mask = aa.mask.circular(
    shape_2d=dataset.shape_2d,
    pixel_scales=dataset.pixel_scales,
    sub_size=1,
    radius=10.0,
)

# Its also worth checking out the 'fit.py' module in this tutorial. The module follows the same structure as tutorials
# 2 and 3, but the functions has been updated to include the use of a mask.

# We're now going to perform multiple fits, where each fit changes different aspects of how the fit is performed.
# To do this, we'll set up phase with a `phase-setting', the signal_to_noise_limit.

# - signal_to_noise_limit:

#       The dataset's image and noise-map give an image with a specific signal-to-noise ratio in every pixel. By
#       inputting a signal_to_noise_limit the dataset's noise map is scaled such that all pixels above the input
#       signal-to-noise threshold have a signal-to-noise at the input threshold.

# (For the purpose of learning PyAutoFit details of what the phase setting does arn't too important, we're more
# interested in how this is input into a phase, how it changes the analysis and output directory structure).

# For our first phase, we will omit both the phase setting (by setting it to None) and reperform the fit of tutorial 3.

phase = ph.Phase(
    phase_name="phase_t4",
    gaussian=af.PriorModel(gaussians.Gaussian),
    signal_to_noise_limit=None,
)

print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t4"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# Okay, lets think about what was performed differently in this phase. To begin, lets note the output directory:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_4_phase_features/output/phase_example/phase_tag'

# There is a small change to this directory compared to tutorial 3, there is a new folder 'phase_tag' within which the
# results are stored. It'll be clear why this is in a moment.

# Next, lets check that this phase did indeed perform visualization. Navigate to the folder 'image' in the directory
# above. You should now see a set of folders containing visualization of the dataset and fit, as well as a subplots
# folder. As promised, our phase is now taking care of the visualization of our model.

# This visualization happens 'on-the-fly', such that during a MultiNest these images are output using the current
# best-fit model MultiNest has found. For models more complex than our 2D Gaussian this is useful, as it means we can
# often check that MultiNest has found reasonable solutions during a run and can thus cancel it early if it has ended up
# with an incorrect solution.

# How often does PyAutoFit output new images? This is set by the 'visualize_interval' in the config file
# 'config/visualize/general.ini'

# Next, we're going to customize and run a phase using the signal_to_noise_limit. To do this, we create a new phase and
# run it as per usual, but additionally inputting a signal_to_noise_limit of 10.0.

phase = ph.Phase(
    phase_name="phase_example",
    gaussian=af.PriorModel(gaussians.Gaussian),
    signal_to_noise_limit=10.0,
)

phase.run(dataset=dataset, mask=mask)

# You'll note the results are now in a slightly different directory to the fit performed above:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_4_phase_features/output/phase_example/phase_tag__snr_10.0'

# By customizing ou phase's settings, PyAutoFit has changed it output path using a tag for this phase. There are two
# reasons PyAutoFit does this:

# 1) Tags describes the analysis settings, making it explicit what analysis was used to create the results.

# 2) Tags create a unique output path, ensuring that if you run multiple phases on the same data but with different
#    settings each non-linear search (e.g. MultiNest) won't inadvertantly use results generated via a different analysis
#    method.


# To perform a non-linear search in PyAutoFit we use Phase objects. A Phase performs the following tasks:

# - Builds the model to be fitted and interfaces it with the non-linear search algorithm.
# - Receives the data to be fitted and prepares it so the model can fit it.
# - When the non-linear search is running, defines the function that enables a likelihood to be computed given a model
#   instance.
# - Handles Visualization of the results, albeit this feature will be omitted until the next tutorial.
# - Returns results giving the best-fit model and the inferred parameters (with errors) of the models fit to the data.

# At this point, you should open and inspect (in detail) the files 'phase.py', 'analysis.py' and 'result.py'. These
# 3 files are the heart of any PyAutoFit model fit - these are the only files you need in order fit any model
# imaginable to any data-set imaginable! An over view of each is as follows:

# phase.py:

#   - Receives the model to be fitted (in this case a single Gaussian).
#   - Handles the directorry structure of the output (in this example results are output to the folder
#     'autofit_workspace/howtofit/chapter_1_introduction/tutorial_3_non_linear_search/output/phase_example/'.
#   - Is passed the data when run, which is then set up for the analsyis.

# analysis.py:

#   - Prepares the dataset for fitting (e.g. masking).
#   - Fits this dataset with a model instance to compute a likelihood for every iteration of the non-linear search.
#   - Handles visualization (this is omitted until tutorial 4)!

# result.py

#   - Stores he best-fit (highest likelhood) model instance.
#   - Has functions to create the best-fit model image, best-fit residuals, etc.
#   - Has functions to inspect the overall quality of the model-fit (e.g. parameter estimates, errors, etc.). These
#     will be detailed in chapter 5.

# Once the above files are set up correct, performing a model-fit in PyAutoFit boils down to two lines of code, simply
# making the phase (given a model) and running the phase (by passing it data). Go ahead and do it!
