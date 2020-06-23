from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np

from autoconf import conf
import autofit as af
from autofit_workspace.examples.complex import model as m
from autofit_workspace.examples.complex import analysis as a

# %%
"""
In this example, we'll fit 1D data of a Gaussian and Exponential profile with a 1D Gaussian + Exponential model using 
the non-linear searches emcee and Dynesty.

If you haven't already, you should checkout the files 'example/model.py' and 'example/analysis.py' to see how we have
provided PyAutoFit with the necessary information on our model, data and log likelihood function.
"""

# %%
#%matplotlib inline


# %%
"""
__Paths__

Setup the path to the autofit_workspace, using a relative directory name.
"""

# %%
workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

# %%
"""
Use this path to explicitly set the config path and output path.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""
__Data__

First, lets load our data of a 1D Gaussian + Exponential, which we'll plot before we perform the model-fit.
"""

# %%
dataset_path = f"{workspace_path}/dataset/gaussian_x1__exponential_x1"

data_hdu_list = fits.open(f"{dataset_path}/data.fits")
data = np.array(data_hdu_list[0].data)

noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
noise_map = np.array(noise_map_hdu_list[0].data)

plt.plot(range(data.shape[0]), data)
plt.show()
plt.close()

# %%
"""
__Model__

Next, we create our model, which in this case corresponds to a Gaussian + Exponential. In model.py, you will have
noted the Gaussian has 3 parameters (centre, intensity and sigma) and Exponential 3 parameters (centre, intensity and
rate). These are the free parameters of our model that the non-linear search fits for, meaning the non-linear
parameter space has dimensionality = 6.

In the simple example tutorial, we used a PriorModel to create the model of the Gaussian. PriorModels cannot be used to
compose models from multiple model components and for this example we must instead use the CollectionPriorModel.
"""

# %%
model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

# %%
"""
We can again customize the priors of our model used by the non-linear search.
"""

# %%
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

# %%
"""
Above, we named our model-components: we called the Gaussian component 'gaussian' and Exponential component
'exponential'. We could have chosen anything for these names, as shown by the code below.
"""

# %%
model_custom_names = af.CollectionPriorModel(rich=m.Gaussian, james=m.Exponential)

model_custom_names.rich.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model_custom_names.rich.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model_custom_names.rich.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model_custom_names.james.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model_custom_names.james.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model_custom_names.james.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

# %%
"""
The naming of model components is important - as these names will are adotped by the instance pass to the Analysis
class and the results returned by the non-linear search.

We'll use the 'model' variable from here on, with the more sensible names of 'gaussian' and 'exponential'.
"""

# %%
"""
__Analysis__

We now set up our Analysis, using the class described in 'analysis.py'. The analysis describes how given an instance
of our model (a Gaussian + Exponential) we fit the data and return a log likelihood value. For this simple example,
we only have to pass it the data and its noise-map.
"""

# %%
analysis = a.Analysis(data=data, noise_map=noise_map)

# %%
"""
#####################
###### DYNESTY ######
#####################

We finally choose and set up our non-linear search. We'll first fit the data with the nested sampling algorithm
Dynesty. Below, we manually specify all of the Dynesty settings, however if we omitted them the default values
found in the config file 'config/non_linear/Dynesty.ini' would be used.

For a full description of Dynesty checkout its Github and documentation webpages:

https://github.com/joshspeagle/dynesty
https://dynesty.readthedocs.io/en/latest/index.html
"""

# %%
dynesty = af.DynestyStatic(
    n_live_points=60,
    bound="multi",
    sample="auto",
    bootstrap=0,
    enlarge=-1,
    update_interval=-1.0,
    vol_dec=0.5,
    vol_check=2.0,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    iterations_per_update=500,
    number_of_cores=1,
    paths=af.Paths(folders=["examples", "complex"]),
)

# %%
"""
To perform the fit with Dynesty, we pass it our model and analysis and we're good to go!

Checkout the folder 'autofit_workspace/output/dynestystatic', where the non-linear search results, visualizaion and
information can be found.
"""

# %%
result = dynesty.fit(model=model, analysis=analysis)

# %%
"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood Gaussian + Exponential model to the data.
"""

# %%
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.line_from_xvalues(xvalues=np.arange(data.shape[0]))
model_exponential = instance.exponential.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D Gaussian + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

# %%
"""
We discuss in more detail how to use a results object in the files 'autofit_workspace/example/results'.
"""

# %%
"""
#################
##### Emcee #####
#################

To use a different non-linear we simply use call a different search from PyAutoFit, passing it the same the model
and analysis as we did before to perform the fit. Below, we fit the same dataset using the MCMC sampler Emcee.
Again, we manually specify all of the Emcee settings, however if they were omitted the values found in the config
file 'config/non_linear/Emcee.ini' would be used instead.

For a full description of Emcee, checkout its Github and readthedocs webpages:

https://github.com/dfm/emcee
https://emcee.readthedocs.io/en/stable/

**PyAutoFit** extends **emcee** by providing an option to check the auto-correlation length of the samples
during the run and terminating sampling early if these meet a specified threshold. See this page
(https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description of how this is implemented.
"""

# %%
emcee = af.Emcee(
    nwalkers=50,
    nsteps=2000,
    initialize_method="ball",
    initialize_ball_lower_limit=0.49,
    initialize_ball_upper_limit=0.51,
    auto_correlation_check_for_convergence=True,
    auto_correlation_check_size=100,
    auto_correlation_required_length=50,
    auto_correlation_change_threshold=0.01,
    number_of_cores=1,
    paths=af.Paths(folders=["examples", "complex"]),
)

# %%
result = emcee.fit(model=model, analysis=analysis)

# %%
"""
__Result__

The result object returned by Emcee's fit is similar in structure to the Dynesty result above - it again provides
us with the maximum log likelihood instance.
"""

# %%
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.line_from_xvalues(xvalues=np.arange(data.shape[0]))
model_exponential = instance.exponential.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D Gaussian + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.savefig(f"{workspace_path}/toy_model_fit_x2.png", bbox_inches="tight")
plt.show()
plt.close()

# %%
"""
############################
###### PARTICLE SWARM ######
############################

PyAutoFit also supports a number of searches, which seem to find the global (or local) maxima likelihood solution.
Unlike nested samplers and MCMC algorithms, they do not extensive map out parameter space. This means they can find the
best solution a lot faster than these algorithms, but they do not properly quantify the errors on each parameter.

We'll use the Particle Swarm Optimization algorithm PySwarms. For a full description of PySwarms, checkout its Github 
and readthedocs webpages:

https://github.com/ljvmiranda921/pyswarms
https://pyswarms.readthedocs.io/en/latest/index.html

**PyAutoFit** extends *PySwarms* by allowing runs to be terminated and resumed from the point of termination, as well
as providing different options for the initial distribution of particles.

"""

# %%
pso = af.PySwarmsGlobal(
    n_particles=30,
    iters=1000,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    initialize_method="prior",
    initialize_ball_lower_limit=0.49,
    initialize_ball_upper_limit=0.51,
    number_of_cores=1,
    paths=af.Paths(folders=["examples", "complex"]),
)
result = pso.fit(model=model, analysis=analysis)

# %%
"""
__Result__

The result object returned by PSO is again very similar in structure to previous results.
"""

# %%
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.line_from_xvalues(xvalues=np.arange(data.shape[0]))
model_exponential = instance.exponential.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D Gaussian + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.savefig(f"{workspace_path}/toy_model_fit_x2.png", bbox_inches="tight")
plt.show()
plt.close()


# %%
"""
##########################
##### Other Samplers #####
##########################

Checkout ? for all of the non-linear searches available in PyAutoFit.
"""
