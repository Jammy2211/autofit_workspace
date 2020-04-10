#%%
"""
__Non-linear Search__

Okay, so its finally time to take our model and fit it to our data, hurrah!

So, how do we infer the parameters for a Gaussian that give a good fit to our dataset?  In the last tutorial, we
tried a very basic approach, randomly guessing models until we found one that gave a good fit and high likelihood.

We discussed that this wasn't really a viable strategy for more complex models, and it isn't. However, this is the
basis of how model fitting actually works! Basically, our model-fitting algorithm guesses lots of models, tracking
the likelihood of these models. As the algorithm progresses, it begins to guess more models using parameter
combinations that gave higher likelihood solutions previously. If a set of parameters provided a good fit to the
dataset previously, a model with similar values probably will too.

This is called a 'non-linear search' and its a fairly common problem faced by scientists. We're going to use a
non-linear search algorithm called 'MultiNest'. For now, lets not worry about the details of how MultiNest actually
works. Instead, just picture that a non-linear search in PyAutoFit operates as follows:

1) Randomly guess models, mapping their parameters via priors to instances of the model, in this case a Gaussian.

2) Use this model instance to generate model data and compare this model data to the dataset to compute a likelihood.

3) Repeat this many times, using the likelihoods of previous fits (typically those with a high likelihood) to
   find models with higher likelihoods.

In chapter 2, we'll go into the details of how a non-linear search works and outline the benefits and drawbacks of
different non-linear search algorithms. In this chapter, we just want to convince ourselves that we can fit a model!
"""


# %%
#%matplotlib inline

# %%
import autofit as af

from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.plot import (
    fit_plots,
)
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.model import (
    gaussian,
)
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.phase import (
    phase as ph,
)

# %%
"""
You need to change the path below to the chapter 1 directory so we can load the dataset.
"""

# %%
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
af.conf.instance = af.conf.Config(
    config_path=chapter_path + "/config",
    output_path=chapter_path
    + "output",  # <- This sets up where the non-linear search's outputs go.
)

dataset_path = chapter_path + "dataset/gaussian_x1/"

from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    data_path=dataset_path + "data.fits", noise_map_path=dataset_path + "noise_map.fits"
)

# %%
"""
To perform a non-linear search in PyAutoFit we use Phase objects. A Phase performs the following tasks:

- Builds the model to be fitted and interfaces it with the non-linear search algorithm.
- Receives the data to be fitted and prepares it so the model can fit it.
- When the non-linear search is running, defines the function computes a likelihood given a model instance.
- Returns results giving the best-fit model and the inferred parameters (with errors) of the models fit to the data.

We'll look at how the phase is set up in the tutorial in a moment, but first lets instantiate and run a phase to.
Performing a model-fit in PyAutoFit boils down to two lines of code, simply making the phase (specifying a model)
and running the phase (by passing it data). Go ahead and do it!
"""

# %%
phase = ph.Phase(phase_name="phase_t3", gaussian=af.PriorModel(gaussian.Gaussian))

# %%
"""
This line will set off the non-linear search MultiNest - it'll probably take a minute or so to run (which is very
fast for a model-fit). Whilst you're waiting, checkout the folder:

'autofit_workspace/howtofit/chapter_1_introduction/output/phase_t3/'

Here, the results of the model-fit are output to your hard-disk on-the-fly and you can inspect them as the non-linear
search runs. In particular, you'll file:

- model.info: A file listing every model component, parameter and prior in your model-fit.
- model.results: A file giving the latest best-fit model, parameter estimates and errors of the fit.
- optimizer: A folder containing the MultiNest output .txt files (you'll probably never need to look at these, but
         its good to know what they are).
- Other metadata which you can ignore for now.
"""

# %%
print(
    "MultiNest has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t3"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take a few minutes!"
)

result = phase.run(dataset=dataset)

print("MultiNest has finished run - you may now continue the notebook.")

# %%
"""
Once complete, the phase results a Result object, which as mentioned contains the best-fit model instance.
"""

# %%
print("Best-fit Model:\n")
print("Centre = ", result.instance.gaussian.centre)
print("Intensity = ", result.instance.gaussian.intensity)
print("Sigma = ", result.instance.gaussian.sigma)

# %%
"""
The Result class also has functions which generate an instance of the fit class using the best-fit model.
"""

# %%
fit_plots.model_data(fit=result.most_likely_fit)
fit_plots.residual_map(fit=result.most_likely_fit)
fit_plots.chi_squared_map(fit=result.most_likely_fit)

# %%
"""
We also have an 'output' attribute, which in this case is a MultiNestOutput object:
"""

# %%
print(result.output)

# %%
"""
This object acts as an interface between the MultiNest output results on your hard-disk and this Python code. For
example, we can use it to get the evidence estimated by MultiNest.
"""

# %%
print(result.output.evidence)

# %%
"""
We can also use it to get a model-instance of the "most probable" model, which is the model where each parameter is
the value estimated from the probability distribution of parameter space.
"""

# %%
mp_instance = result.output.most_probable_instance
print()
print("Most Probable Model:\n")
print("Centre = ", mp_instance.gaussian.centre)
print("Intensity = ", mp_instance.gaussian.intensity)
print("Sigma = ", mp_instance.gaussian.sigma)

# %%
"""
We'll come back to this output object in tutorial 7!

At this point, you should open and inspect (in detail) the source code files 'phase.py', 'analysis.py' and 'result.py'.
These 3 files are the heart of any PyAutoFit model fit - they are the only files you need in order fit a model to a
data-set! An over view of each is as follows:

phase.py:

- Receives the model to be fitted (in this case a single Gaussian).
- Handles the directory structure of the output (in this example results are output to the folder
'/output/phase_example/'.
- Is passed the data when run, which is set up for the analysis.

analysis.py:

- Prepares the dataset for fitting.
- Fits this dataset with a model instance to compute a likelihood for every iteration of the non-linear search.

result.py

- Stores the best-fit (highest likelihood) model instance.
- Has functions to create the best-fit model image, best-fit residuals, etc.
- Has functions to inspect the overall quality of the model-fit (e.g. parameter estimates, errors, etc.). These
will be detailed in chapter 5.

Finally, the other thing to think about is the directory structure of the tutorial's 'source code', where we have
separated modules into 5 packages: 'dataset', 'fit', 'model', 'plot' and 'phase'. This cleanly separates different
parts of the code which do different thing and is a design I recommend your model-fitting project strictly adheres to!

For example, this ensures the code which handles the model is completely separate from the code which handles phases.
The model then never interfaces directly with PyAutoFit, ensuring good code design by removing dependencies between
parts of the code that do not need to interact! Its the same for the part of the code that stores data ('dataset')
and fits a model to a dataset ('fit') - by keeping them separate its clear which part of the code do what task.

This is a principle aspect of object oriented design and software engineering called 'separation of concerns' and all
templates we provide in the HowToFit series will adhere to it.
"""
