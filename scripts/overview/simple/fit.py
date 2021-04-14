"""
__Example: Fit__

In this example, we'll fit 1D data of a `Gaussian` profile with a 1D `Gaussian` model using the non-linear searches
emcee, Dynesty and PySwarms.

If you haven't already, you should checkout the files `example/model.py` and `example/analysis.py` to see how we have
provided PyAutoFit with the necessary information on our model, data and log likelihood function.
"""
import autofit as af
import model as m
import analysis as a

import matplotlib.pyplot as plt
import numpy as np
from os import path

"""
__Data__

First, lets load data of a 1D Gaussian, by loading it from a .json file in the directory 
`autofit_workspace/dataset//gaussian_x1`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Now lets plot the data, including its error bars. We'll use its shape to determine the xvalues of the
data for the plot.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.show()
plt.close()

"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian. In model.py, you will have noted
this `Gaussian` has 3 parameters (centre, intensity and sigma). These are the free parameters of our model that the
non-linear search fits for, meaning the non-linear parameter space has dimensionality = 3.
"""
model = af.PriorModel(m.Gaussian)

"""
Checkout `autofit_workspace/config/priors/model.json`, this config file defines the default priors of the `Gaussian` 
model component. 

We can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.intensity = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We now set up our Analysis, using the class described in `analysis.py`. The analysis describes how given an instance
of our model (a Gaussian) we fit the data and return a log likelihood value. For this simple example, we only have to
pass it the data and its noise-map.
"""
analysis = a.Analysis(data=data, noise_map=noise_map)

"""
#############################
###### NESTED SAMPLING ######
#############################

We finally choose and set up our non-linear search. we'll first fit the data with the nested sampling algorithm
Dynesty. Below, we manually specify all of the Dynesty settings, however if we omitted them the default values
found in the config file `config/non_linear/Dynesty.ini` would be used.

We also specify a `path_prefix`, so that our results go to the folder `autofit_workspace/output/overview/simple`.

For a full description of Dynesty checkout its Github and documentation webpages:

https://github.com/joshspeagle/dynesty
https://dynesty.readthedocs.io/en/latest/index.html
"""
dynesty = af.DynestyStatic(
    path_prefix=path.join("overview", "simple", "fit"),
    nlive=100,
    bound="multi",
    sample="auto",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    vol_dec=0.5,
    vol_check=2.0,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    iterations_per_update=500,
    number_of_cores=1,
)

"""
To perform the fit with Dynesty, we pass it our model and analysis and we`re good to go!

Checkout the folder `autofit_workspace/output/dynestystatic`, where the `NonLinearSearch` results, visualization and
information can be found.
"""
result = dynesty.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Dynesty model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""
We discuss in more detail how to use a results object in the files `autofit_workspace/example/simple/result.py`.

##################
###### MCMC ######
##################

To use a different non-linear we simply use call a different search from PyAutoFit, passing it the same the model
and analysis as we did before to perform the fit. Below, we fit the same dataset using the MCMC sampler Emcee.
Again, we manually specify all of the Emcee settings, however if they were omitted the values found in the config
file `config/non_linear/Emcee.ini` would be used instead.

For a full description of Emcee, checkout its Github and readthedocs webpages:

https://github.com/dfm/emcee
https://emcee.readthedocs.io/en/stable/

**PyAutoFit** extends **emcee** by providing an option to check the auto-correlation length of the samples
during the run and terminating sampling early if these meet a specified threshold. See this page
(https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description of how this is implemented.
"""
emcee = af.Emcee(
    path_prefix=path.join("overview", "simple", "fit"),
    nwalkers=30,
    nsteps=1000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    iterations_per_update=500,
    number_of_cores=1,
)
result = emcee.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by Emcee`s fit is similar in structure to the Dynesty result above - it again provides
us with the maximum log likelihood instance.
"""
model_data = result.max_log_likelihood_instance.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Emcee model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""
############################
###### PARTICLE SWARM ######
############################

PyAutoFit also supports a number of searches, which seem to find the global (or local) maxima likelihood solution.
Unlike nested samplers and MCMC algorithms, they do not extensive map out parameter space. This means they can find the
best solution a lot faster than these algorithms, but they do not properly quantify the errors on each parameter.

we'll use the Particle Swarm Optimization algorithm PySwarms. For a full description of PySwarms, checkout its Github 
and readthedocs webpages:

https://github.com/ljvmiranda921/pyswarms
https://pyswarms.readthedocs.io/en/latest/index.html

**PyAutoFit** extends *PySwarms* by allowing runs to be terminated and resumed from the point of termination, as well
as providing different options for the initial distribution of particles.

"""
pso = af.PySwarmsGlobal(
    path_prefix=path.join("overview", "simple", "fit"),
    n_particles=50,
    iters=100,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    initializer=af.InitializerPrior(),
    number_of_cores=1,
)
result = pso.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by PSO is again very similar in structure to previous results.
"""
model_data = result.max_log_likelihood_instance.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("PySwarms model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""
__Other Samplers__

Checkout https://pyautofit.readthedocs.io/en/latest/api/api.html for the non-linear searches available in PyAutoFit.
"""
