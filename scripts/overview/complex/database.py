"""
__Example: Fit__

In this example, we'll fit 1D data of a `Gaussian` and Exponential profile with a 1D `Gaussian` + Exponential model using
the non-linear searches Emcee and Dynesty.

If you haven't already, you should checkout the files `example/model.py` and `example/analysis.py` to see how we have
provided PyAutoFit with the necessary information on our model, data and log likelihood function.
"""
# %matplotlib inline

from os import path

import matplotlib.pyplot as plt
import numpy as np

import analysis as a
import autofit as af
import model as m

"""
__Data__

First, lets load data of a 1D `Gaussian` + 1D Exponential, by loading it from a .json file in the directory 
`autofit_workspace/dataset//gaussian_x1__exponential_x1`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

session = af.db.open_database(
    "database.sqlite"
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

Next, we create our model, which in this case corresponds to a `Gaussian` + Exponential. In model.py, you will have
noted the `Gaussian` has 3 parameters (centre, intensity and sigma) and Exponential 3 parameters (centre, intensity and
rate). These are the free parameters of our model that the `NonLinearSearch` fits for, meaning the non-linear
parameter space has dimensionality = 6.

In the simple example tutorial, we used a `PriorModel` to create the model of the Gaussian. PriorModels cannot be used to
compose models from multiple model components and for this example we must instead use the CollectionPriorModel.
"""
model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

"""
Checkout `autofit_workspace/config/priors/model.json`, this config file defines the default priors of the `Gaussian` 
and `Exponential` model components. 

We can manually customize the priors of our model used by the non-linear search.
"""
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
Above, we named our model-components: we called the `Gaussian` component `gaussian` and Exponential component
`exponential`. We could have chosen anything for these names, as shown by the code below.
"""
model_custom_names = af.CollectionPriorModel(
    custom_name=m.Gaussian, another_custom_name=m.Exponential
)

model_custom_names.custom_name.centre = af.UniformPrior(
    lower_limit=0.0, upper_limit=100.0
)
model_custom_names.custom_name.intensity = af.UniformPrior(
    lower_limit=0.0, upper_limit=1e2
)
model_custom_names.custom_name.sigma = af.UniformPrior(
    lower_limit=0.0, upper_limit=30.0
)
model_custom_names.another_custom_name.centre = af.UniformPrior(
    lower_limit=0.0, upper_limit=100.0
)
model_custom_names.another_custom_name.intensity = af.UniformPrior(
    lower_limit=0.0, upper_limit=1e2
)
model_custom_names.another_custom_name.rate = af.UniformPrior(
    lower_limit=0.0, upper_limit=10.0
)

"""
The naming of model components is important - as these names will are adotped by the instance pass to the Analysis
class and the results returned by the non-linear search.

we'll use the `model` variable from here on, with the more sensible names of `gaussian` and `exponential`.

__Analysis__

We now set up our Analysis, using the class described in `analysis.py`. The analysis describes how given an instance
of our model (a `Gaussian` + Exponential) we fit the data and return a log likelihood value. For this simple example,
we only have to pass it the data and its noise-map.
"""
analysis = a.Analysis(data=data, noise_map=noise_map)

"""
#####################
###### DYNESTY ######
#####################

We finally choose and set up our non-linear search. we'll first fit the data with the nested sampling algorithm
Dynesty. Below, we manually specify all of the Dynesty settings, however if we omitted them the default values
found in the config file `config/non_linear/Dynesty.ini` would be used.

For a full description of Dynesty checkout its Github and documentation webpages:

https://github.com/joshspeagle/dynesty
https://dynesty.readthedocs.io/en/latest/index.html
"""
dynesty = af.DynestyStatic(
    path_prefix=path.join("overview", "complex", "fit"),
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
    session=session
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
compare the maximum log likelihood `Gaussian` + Exponential model to the data.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""
We discuss in more detail how to use a results object in the files `autofit_workspace/example/results`.

#################
##### Emcee #####
#################

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
    path_prefix=path.join("overview", "complex", "fit"),
    nwalkers=50,
    nsteps=2000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlation_check_for_convergence=True,
    auto_correlation_check_size=100,
    auto_correlation_required_length=50,
    auto_correlation_change_threshold=0.01,
    number_of_cores=1,
    session=session
)

result = emcee.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by Emcee`s fit is similar in structure to the Dynesty result above - it again provides
us with the maximum log likelihood instance.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Emcee model fit to 1D Gaussian + Exponential dataset.")
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
pso = af.PySwarmsLocal(
    path_prefix=path.join("overview", "complex", "fit"),
    n_particles=100,
    iters=1000,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    initializer=af.InitializerPrior(),
    number_of_cores=1,
    session=session
)
result = pso.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by PSO is again very similar in structure to previous results.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.profile_from_xvalues(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("PySwarms model fit to 1D Gaussian + Exponential dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile intensity")
plt.show()
plt.close()

"""
__Other Samplers__

Checkout https://pyautofit.readthedocs.io/en/latest/api/api.html for the non-linear searches available in PyAutoFit.
"""
