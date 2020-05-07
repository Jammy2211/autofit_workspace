import autofit as af
from astropy.io import fits
import os
import matplotlib.pyplot as plt

from autofit_workspace.api.simple import model as m
from autofit_workspace.api.simple import analysis as a

import numpy as np

# In this example, we'll fit 1D data of a Gaussian profile with a 1D Gaussian model using the non-linear search
# MulitNest.

# If you haven't already, you should checkout the files 'api/model.py' and 'api/analysis.py' to see how we have
# provided PyAutoFit with the necessary information on our model, data and log likelihood function.

### Data ###

# First, lets load our data of a 1D Gaussian, whcih we'll plot before perform the model-fit.

dataset_path = "{}/../../dataset/gaussian_x1".format(
    os.path.dirname(os.path.realpath(__file__))
)

data_hdu_list = fits.open(f"{dataset_path}/data.fits")
data = np.array(data_hdu_list[0].data)

noise_map_hdu_list = fits.open(f"{dataset_path}/noise_map.fits")
noise_map = np.array(noise_map_hdu_list[0].data)

plt.plot(range(data.shape[0]), data)
plt.show()
plt.close()

### Model ###

# Next, we create our model, which in this case correspoonds to a single Gaussian. In model.py, you will have noted
# this Gaussian has 3 parameters (centre, intensity and sigma). These are the free parameters of our model that the
# non-linear search fits for.

model = af.PriorModel(m.Gaussian)

# Checkout 'autofit_workspace/config/json_priors' - this config file defines the default priors of all our model
# components. However, we can overwrite priors before running the non-linear search as shown below.

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.LogUniformPrior(lower_limit=1e-1, upper_limit=1e6)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

### Analysis ###

# We now set up our Analysis, using the class described in 'analysis.py'. The analysis describes how given an instance
# of our model (a Gaussian) we fit the data and return a log likelihood value. For this simple example, we only have to
# pass it the data and its noise-map.

analysis = a.Analysis(data=data, noise_map=noise_map)

#####################
##### MULTINEST #####
#####################

# We finally choose and set up our non-linear search. First, lets fit the data with the nested sampling algorithm
# MulitNest. Below, we manually specify all of the MultiNest settings, however if we chose to omit them they would use
# the default values found in the config file 'config/non_linear/MultiNest.ini'.

# For a full description of MultiNest and its Python wrapper PyMultiNest, checkout its Github and documentation
# webpages:
#
# https://github.com/JohannesBuchner/MultiNest
# https://github.com/JohannesBuchner/PyMultiNest
# http://johannesbuchner.github.io/PyMultiNest/index.html#

multi_nest = af.MultiNest(
    n_live_points=50,
    sampling_efficiency=0.5,
    const_efficiency_mode=False,
    evidence_tolerance=0.8,
    multimodal=True,
    importance_nested_sampling=False,
    n_iter_before_update=5,
    null_log_evidence=-1.0e90,
    max_modes=100,
    mode_tolerance=-1e90,
    seed=-1,
    verbose=False,
    resume=True,
    context=0,
    write_output=True,
    log_zero=-1e100,
    max_iter=0,
    init_MPI=True,
)

# To perform the fit with MultiNest, we pass it our model and analysis and we're good to go!

# Checkout the folder 'autofit_workspace/output/multinest', where the non-linear search results, visualizaion and
# information can be found.

result = multi_nest.fit(model=model, analysis=analysis)

### Result ###

# The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
# compare the maximum log likelihood Gaussian to the data.

model_data = result.max_log_likelihood_instance.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.show()
plt.close()

# We discuss in more detail how to use a results object in the files 'autofit_workspace/api/results'.

#################
##### Emcee #####
#################

#  Using a different non-linear search is easy, we simply call a different search from PyAutoFit and pass it the same
#  the model and analysis to perform the fit. Below, we fit the same dataset using the MCMC sampler Emcee. Again, we
#  manually specify all of the Emcee settings, however if they were omitted the values found in the config file
#  'config/non_linear/Emcee.ini' would be used instead.

# For a full description of Emcee, checkout its Github and readthedocs webpages:

# https://github.com/dfm/emcee
# https://emcee.readthedocs.io/en/stable/

# **PyAutoFit** extends **emcee** by providing an option to check the auto-correlation length of the samples
# during the run and terminating sampling early if these meet a specified threshold. See this page
# (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description of how this is implemented.

emcee = af.Emcee(
    nwalkers=10,
    nsteps=10,
    initialize_method="ball",
    auto_correlation_check_for_convergence=True,
    auto_correlation_check_size=100,
    auto_correlation_required_length=50,
    auto_correlation_change_threshold=0.01,
)
result = emcee.fit(model=model, analysis=analysis)

### Result ###

# The result object returned by Emcee's fit is similar in structure to the MultiNest result above - it again provides
# us with the maximum log likelihood instance.

model_data = result.max_log_likelihood_instance.line_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.show()
plt.close()

##########################
##### Other Samplers #####
##########################

# Checkout ? for all of the non-linear searches available in PyAutoFit.
