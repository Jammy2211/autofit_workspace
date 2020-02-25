import autofit as af
import autoarray as aa
import autoarray.plot as aplt

from howtofit.chapter_1_introduction.tutorial_3_non_linear_search import (
    gaussians,
    phase as ph,
)

# Okay, so its finally time to take our model and fit it to our data, hurrah!

# So, how do we infer the parameters for a Gaussian that that give a good fit to our dataset?  In the last tutorial, we
# tried a very basic approach, randomly guessing models until we found one that gave a good fit and high likelihood.

# We discussed that this wasn't really a viable strategy for more complex models, and it isn't. Surprisisngly, however,
# this is the basis of how model fitting actually works! Basically, our model-fitting algorithm guesses lots of models,
# tracking the likelihood of these models. As the algorithm progresses, it begins to guess more models using parameter
# combinations that gave higher likelihood solutions previously, with the idea that if a set of parameters provided a
# good fit to the dataset, a model with similar values probably will too.

# This is called a 'non-linear search' and its a fairly common problem faced by scientists. We're going to use a
# non-linear search algorithm called 'MultiNest'. For now, lets not worry about the details of how MultiNest actually
# works. Instead, just picture that a non-linear search in PyAutoFit operates as follows:

# 1) Randomly guess a model and map its parameters via the priors to make an instance of a Gaussian.

# 2) Use this model instance to generate a model image and compare this model image to the imaging data to compute
#    a likelihood.

# 3) Repeat this many times, using the likelihoods of previous fits (typically those with a high likelihood) to
#    guide us to the lens models with the highest likelihood.

# In chapter 2, we'll go into the details of how a non-linear search works and outline the benefits and drawbacks of
# different non-linear search algorithms. In this chapter, we just want to convince ourselves that we can fit a model!

# You need to change the path below to the chapter 1 directory so we can load the dataset#
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config",
    output_path=chapter_path
    + "tutorial_3_non_linear_search/output",  # <- This sets up where the non-linear search's outputs go.
)

dataset_path = chapter_path + "dataset/gaussian_x1/"

imaging = aa.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    pixel_scales=1.0,
)

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

phase = ph.Phase(phase_name="phase_example", gaussian=af.PriorModel(gaussians.Gaussian))

# This line will set off the non-linear search MultiNest - it'll probably take a minute or so to run (which is very
# fast for a model-fit). Whilst you're waiting, checkout the folder:

# 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_3_non_linear_search/output/phase_example/'

# Here, the results of the model-fit are output to your hard-disk on-the-fly and you can inspect them as the non-linear
# search runs. In particular, you'll file:

# - model.info: A file listing every model component, parameter and prior in your model-fit.
# - model.results: A file giving the latest best-fit model, parameter estimates and errors of the fit.
# - optimizer: A folder containing the MultiNest output .txt files (you'll probably never need to look at these, but
#              its good to know what they are).
# - Other metadata which you can ignore for now.

result = phase.run(dataset=imaging)

# Once complete, the phase results a Result object, which as mentioned contains the best-fit model instance.
print("Best-fit Model:\n")
print("Centre = ", result.instance.gaussian.centre)
print("Intensity = ", result.instance.gaussian.intensity)
print("Sigma = ", result.instance.gaussian.sigma)

# In result.py, we also extended the Result class to have functions which generate the best-fit image and fit from
# the best-fit model.
aplt.array(array=result.most_likely_model_image)
aplt.fit_imaging.subplot_fit_imaging(fit=result.most_likely_fit)
