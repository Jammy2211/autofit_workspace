#%%
"""
__Non-linear Search__

Okay, so its finally time to take our model and fit it to data, hurrah!

So, how do we infer the parameters for the 1D Gaussian that give a good fit to our data?  In the last tutorial, we
tried a very basic approach, randomly guessing models until we found one that gave a good fit and high log_likelihood.

We discussed that this wasn't really a viable strategy for more complex models, and it isn't. However, this is the
basis of how model fitting actually works! Basically, our model-fitting algorithm guesses lots of models, tracking
the log likelihood of these models. As the algorithm progresses, it begins to guess more models using parameter
combinations that gave higher log_likelihood solutions previously. If a set of parameters provided a good fit to the
data previously, a model with similar values probably will too.

This is called a 'non-linear search' and its a fairly common problem faced by scientists. We're going to use a
non-linear search algorithm called 'Emcee', which for those familiar with statistic inference is a Markov Chain Monte
Carlo (MCMC) method. For now, lets not worry about the details of how Emcee actually works. Instead, just picture that
a non-linear search in PyAutoFit operates as follows:

    1) Randomly guess a model, mapping their parameters via the priors to instances of the model, in this case a
       Gaussian.

    2) Use this model instance to generate model data and compare this model data to the data to compute a log
       likelihood.

    3) Repeat this many times, choosing models whose parameter values are near those of the model which currently has
       the highest log likelihood value. If the new model's log likelihood is higher than the previous model, new
       models will thus be chosen with parameters nearer this model.

The idea is that if we keep guessing models with higher log-likelihood values, we'll inevitably 'climb' up the gradient
of the log likelihood in parameter space until we eventually hit the highest log likelihood models.

To be clear, this overly simplfied description of an MCMC algorithm is not how the *non-linear search* really works and
omits the details of how our priors inpact our inference or why an MCMC algorithm can provide reliable errors on our
parameter estimates.

A detailed understanding of *non-linear sampling* is not required in chapter 1, however interested users will find
a complete description of *non-linear searches* in chapter 2 of the HowToFit lectures.
"""

# %%
#%matplotlib inline

# %%
from autoconf import conf
import autofit as af
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

# %%
"""
You need to change the path below to the workspace directory so we can load the dataset.
"""

# %%
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace"
chapter_path = f"{workspace_path}/howtofit/chapter_1_introduction"

# %%
"""
The line conf.instance is now used to set up a second property of the configuration:

    - The path to the PyAutoFit output folder, which is where the results of the non-linear search are written to 
      on your hard-disk, alongside visualization and other properties of the fit 
      (e.g. '/path/to/autolens_workspace/output/howtolens')

(These will work autommatically if the WORKSPACE environment variable was set up correctly during installation. 
Nevertheless, setting the paths explicitly within the code is good practise.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config",
    output_path=f"{workspace_path}/output/howtofit",  # <- This sets up where the non-linear search's outputs go.
)

# %%
"""
Lets load the data and noise-map we'll use for our fits, which is the same data we used in tutorial 2.
"""
dataset_path = f"{workspace_path}/dataset/gaussian_x1"

data_hdu_list = fits.open(f"{dataset_path}/data.fits")
data = np.array(data_hdu_list[0].data)

noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
noise_map = np.array(noise_map_hdu_list[0].data)

# %%
"""
Lets remind ourselves what the data looks like, using the plot_lint convenience method fom the previous tutorial.
"""


def plot_line(xvalues, line, ylabel=None):

    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()


xvalues = np.arange(data.shape[0])

plot_line(xvalues=xvalues, line=data, ylabel="Data")

# %%
"""
Lets remake the Gaussian class for this tutorial, which is the model we will fit using the non-linear search.
"""

# %%
class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma

    def line_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the light profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : ndarray
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


# %%
"""
The non-linear search requires an *Analysis* class, which:

    - Receives the data to be fitted and prepares it so the model can fit it.
    - Defines the 'log_likelihood_function' used to compute the log likelihood from a model instance. 
    - Passes this log likelihood to the non-linear search so that it can determine parameter values for the the next model 
      that it samples.

For our 1D Gaussian model-fitting example, here is our *Analysis* class:
"""

# %%
class Analysis(af.Analysis):
    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):

        # The 'instance' that comes into this method is an instance of the Gaussian class above, with the parameters
        # set to values chosen by the non-linear search. (These are commented out to pretent excessive print statements
        # when we run the non-linear search).

        # This instance's parameter values are chosen by the non-linear search based on the previous model with the
        # highest likelihood result.

        # print("Gaussian Instance:")
        # print("Centre = ", instance.centre)
        # print("Intensity = ", instance.intensity)
        # print("Sigma = ", instance.sigma)

        # Below, we fit the data with the Gaussian instance, using its "line_from_xvalues" function to create the
        # model data.

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.line_from_xvalues(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


# %%
"""
To perform the non-linear search using Emcee, we simply compose our model using a PriorModel, instantiate the Analysis
class and pass them to an instance of the Emcee class:
"""

model = af.PriorModel(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

analysis = Analysis(data=data, noise_map=noise_map)

emcee = af.Emcee()

# %%
"""
We begin the non-linear search by calling its fit method, it  will take a minute or so to run (which is very fast for a 
model-fit). Whilst you're waiting, checkout the folder:

'autofit_workspace/howtofit/chapter_1_introduction/output/emcee'

Here, the results of the model-fit are output to your hard-disk (on-the-fly) and you can inspect them as the non-linear
search runs. In particular, you'll fild:

    - model.info: A file listing every model component, parameter and prior in your model-fit.
    - model.results: A file giving the latest best-fit model, parameter estimates and errors of the fit.
    - search: A folder containing the Emcee output in hdf5 format.txt (you'll probably never need to look at these, but
              its good to know what they are).
    - Other metadata which you can ignore for now.
"""

# %%
result = emcee.fit(model=model, analysis=analysis)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t3"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Once completed, the non-linear search returns a Result object, which contains lots of information about the non-linear. 
A full description of the *Results* object can be found at:
 
'autofit_workspace/examples/simple/results'
'autofit_workspace/examples/complex/results'. 

In this tutorial, lets use it to inspect the maximum likelihood model instance.
"""

# %%
print("Maximum Likelihood Model:\n")
max_log_likelihood_instance = result.samples.max_log_likelihood_instance
print("Centre = ", max_log_likelihood_instance.centre)
print("Intensity = ", max_log_likelihood_instance.intensity)
print("Sigma = ", max_log_likelihood_instance.sigma)

# %%
"""
We can use this to plot the maximum log likelihood fit over the data:
"""

# %%
model_data = result.max_log_likelihood_instance.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.xlabel("x")
plt.ylabel("Intensity")
plt.show()
plt.close()

# %%
"""
Above, we used the results 'Samples' property, which in this case is a MCMCSamples object:
"""

# %%
print(result.samples)

# %%
"""
This object acts as an interface between the Emcee output results on your hard-disk and this Python code. For
example, we can use it to get the parameters and log likelihood of every accepted emcee sample.
"""

# %%
print(result.samples.parameters)
print(result.samples.log_likelihoods)

# %%
"""
We can also use it to get a model instance of the "most probable" model, which is the model where each parameter is
the value estimated from the probability distribution of parameter space.
"""

# %%
mp_instance = result.samples.median_pdf_instance
print()
print("Most Probable Model:\n")
print("Centre = ", mp_instance.centre)
print("Intensity = ", mp_instance.intensity)
print("Sigma = ", mp_instance.sigma)

# %%
"""
We'll come back to look at Samples objects in more detail tutorial 8!
"""
