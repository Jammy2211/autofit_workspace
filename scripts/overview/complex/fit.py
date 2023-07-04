"""
__Example: Fit__

In this example, we'll fit 1D data of a `Gaussian` and Exponential profile with a 1D `Gaussian` + Exponential
model using the non-linear searches Emcee and Dynesty.

If you haven't already, you should checkout the files `overview/simple/fit.ipynb` for a basic introduction to
the **PyAutoFit** tools used for model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt

import matplotlib.pyplot as plt
import numpy as np
import os
from os import path

"""
__Data__

Load data of a 1D `Gaussian` + 1D Exponential, by loading it from a .json file in the directory 
`autofit_workspace/dataset//gaussian_x1__exponential_x1`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the data. We'll use its shape to determine the xvalues of the
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

Next, we create our model, which in this case corresponds to a `Gaussian` + Exponential.

We therefore need two classes, one for each model component.
"""


class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """

        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the normalization of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


class Exponential:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        normalization=0.1,  # <- parameters of the Exponential.
        rate=0.01,
    ):
        """Represents a 1D Exponential profile symmetric about a centre, which may be treated as a model-component
        of PyAutoFit the parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        ratw
            The decay rate controlling has fast the Exponential declines.
        """

        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the normalization of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )


"""
We use a `Model` to create and customize the `Gaussian` and `Exponential` models.
"""
gaussian = af.Model(Gaussian)
exponential = af.Model(Exponential)

"""
Checkout `autofit_workspace/config/priors/model.json`, this config file defines the default priors of the `Gaussian` 
and `Exponential` model components. 

We can manually customize the priors of the model used by the non-linear search.
"""
gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
We can now compose the overall model using a `Collection`, which takes the model components we defined above.
"""
model = af.Collection(gaussian=gaussian, exponential=exponential)

"""
The `info` attribute shows the model in a readable format, including the priors specified above.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
Above, we named our model-components: we called the `Gaussian` component `gaussian` and Exponential component
`exponential`. We could have chosen anything for these names, as shown by the code below.
"""
model_custom_names = af.Collection(
    custom_name=gaussian, another_custom_name=exponential
)

print(model_custom_names.custom_name)
print(model_custom_names.another_custom_name)

"""
The naming of model components is important, as these names will are adopted by the instance passed to the `Analysis`
class and the results returned by the non-linear search.

__Analysis__

We now set up our Analysis, which describes how given an instance of our model (a `Gaussian` + Exponential) we fit the
data and return a log likelihood value.

This behaves analogous to the `Analysis` object used in the simple `fit.py` example, but has changes to deal
with the fact that the input model is a `Collection`.
"""


class Analysis(af.Analysis):

    """
    In this example the Analysis only contains the data and noise-map. It can be easily extended however, for more
    complex data-sets and model fitting problems.
    """

    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    """
    In the log_likelihood_function function below, `instance` is an instance of our model, which in this example is
    an instance of the `Gaussian` class and Exponential class in `model.py`. Their parameters are set via the
    non-linear search. This gives us the instance of the model we need to fit our data!
    """

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.Collection
            The model instances of the profiles.

        Returns
        -------
        fit
            The log likelihood value indicating how well this model fit the dataset.
        """

        """
        The `instance` that comes into this method is a Collection. It contains instances of every class
        we instantiated it with, where each instance is named following the names given to the Collection,
        which in this example is a `Gaussian` (with name `gaussian) and Exponential (with name `exponential`):
        """

        # print("Gaussian Instance:")
        # print("Centre = ", instance.gaussian.centre)
        # print("Normalization = ", instance.gaussian.normalization)
        # print("Sigma = ", instance.gaussian.sigma)

        # print("Exponential Instance:")
        # print("Centre = ", instance.exponential.centre)
        # print("Normalization = ", instance.exponential.normalization)
        # print("Rate = ", instance.exponential.rate)

        """Get the range of x-values the data is defined on, to evaluate the model of the profiles."""
        xvalues = np.arange(self.data.shape[0])

        """
        The simplest way to create the summed profile is to add the profile of each model component. If we
        know we are going to fit a `Gaussian` + Exponential we can do the following:

            model_data_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(xvalues=xvalues)
            model_data_exponential = instance.exponential.model_data_1d_via_xvalues_from(xvalues=xvalues)
            model_data = model_data_gaussian + model_data_exponential

        However, this does not work if we change our model components. However, the *instance* variable is a list of
        our model components. We can iterate over this list, calling their model_data_1d_via_xvalues_from and summing the result
        to compute the summed profile of any model.
        """

        """Use these xvalues to create model data of our profiles."""
        model_data = sum(
            [line.model_data_1d_via_xvalues_from(xvalues=xvalues) for line in instance]
        )

        """Fit the model profile data to the observed data, computing the residuals and chi-squareds."""
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search. The `instance` passed
        into the visualize method is maximum log likelihood solution obtained by the model-fit so far and it can be
        used to provide on-the-fly images showing how the model-fit is going.
        """

        xvalues = np.arange(self.data.shape[0])

        model_datas = [
            line.model_data_1d_via_xvalues_from(xvalues=xvalues) for line in instance
        ]
        model_data = sum(model_datas)

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(range(self.data.shape[0]), model_data, color="r")
        for model_data_individual in model_datas:
            plt.plot(range(self.data.shape[0]), model_data_individual, "--")
        plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.clf()


"""
Create the analysis as an instance for the model-fit.
"""
analysis = Analysis(data=data, noise_map=noise_map)

"""
__Paths__

We specify a `path_prefix` which is passed to the non-linear search below, so that our results go to the 
folder `autofit_workspace/output/overview/complex`. The search is also given a `name`, which defines the folder
results are output too.

Results are output to a folder which is a collection of random characters, which is the 'unique_identifier' of
the model-fit. This identifier is generated based on the model fitted and search used, such that an identical
combination of model and search generates the same identifier.

This ensures that rerunning an identical fit will use the existing results to resume the model-fit. In contrast, if
you change the model or search, a new unique identifier will be generated, ensuring that the model-fit results are
output into a separate folder.
"""
path_prefix = path.join("overview", "complex")

"""
#####################
###### DYNESTY ######
#####################

We finally choose and set up our non-linear search. we'll first fit the data with the nested sampling algorithm
Dynesty. Below, we manually specify all of the Dynesty settings, however if we omitted them the default values
found in the config file `config/non_linear/Dynesty.yaml` would be used.

For a full description of Dynesty checkout its Github and documentation webpages:

https://github.com/joshspeagle/dynesty
https://dynesty.readthedocs.io/en/latest/index.html

NOTE: In `autofit_workspace/*/overview/simple/fit.py` we describe how inputting a `name` and `path_prefix` to the
non-linear search outputs the results to hard-disk. 

We do the same below, but checkout that tutorial for an explanation of the benefits of doing this.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="DynestyStatic",
    nlive=60,
    bound="multi",
    sample="rwalk",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    walks=5,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    number_of_cores=1,
)

"""
To perform the fit with Dynesty, we pass it our model and analysis and we`re good to go!

Checkout the folder `autofit_workspace/output/dynestystatic`, where the `NonLinearSearch` results, visualization and
information can be found.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. 

The `info` attribute shows the model in a readable format, including the priors specified above:
"""
print(result.info)

"""
Lets use it to compare the maximum log likelihood `Gaussian` + `Exponential` to the data.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.model_data_1d_via_xvalues_from(
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
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
The Probability Density Functions of the results can be plotted using Dynesty's in-built visualization tools, 
which are wrapped via the `DynestyPlotter` object.
"""
search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
We discuss in more detail how to use a results object in the files `autofit_workspace/example/results`.

#################
##### Emcee #####
#################

To use a different non-linear we simply use call a different search from PyAutoFit, passing it the same the model
and analysis as we did before to perform the fit. Below, we fit the same dataset using the MCMC sampler Emcee.
Again, we manually specify all of the Emcee settings, however if they were omitted the values found in the config
file `config/non_linear/Emcee.yaml` would be used instead.

For a full description of Emcee, checkout its Github and readthedocs webpages:

https://github.com/dfm/emcee
https://emcee.readthedocs.io/en/stable/

**PyAutoFit** extends **emcee** by providing an option to check the auto-correlation length of the samples
during the run and terminating sampling early if these meet a specified threshold. See this page
(https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description of how this is implemented.
"""
search = af.Emcee(
    path_prefix=path_prefix,
    name="Emcee",
    nwalkers=30,
    nsteps=2000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by Emcee`s fit is similar in structure to the Dynesty result above.
 
Printing its `info` shows that it does not estimate a Bayesian evidence, which `dynesty` above did, because it is an
MCMC algoirithm.
"""
print(result.info)

"""
It again provides us with the maximum log likelihood instance which we can use to visualize the fit to the data.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.model_data_1d_via_xvalues_from(
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
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization 
tool `corner.py`, which is wrapped via the `EmceePlotter` object.
"""
search_plotter = aplt.EmceePlotter(samples=result.samples)
search_plotter.corner()

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
search = af.PySwarmsLocal(
    path_prefix=path_prefix,
    name="PySwarmsLocal",
    n_particles=100,
    iters=1000,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    initializer=af.InitializerPrior(),
    number_of_cores=1,
)
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by PSO is again very similar in structure to previous results.

The result info shows that the PSO does not estimate errors on the parameters, because it is a a maximum likelihood
estimator (MLE).
"""
print(result.info)

"""
It again provides us with the maximum log likelihood instance.
"""
instance = result.max_log_likelihood_instance

model_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = instance.exponential.model_data_1d_via_xvalues_from(
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
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
The results can be plotted using the PySwarm's in-built visualization tools which are wrapped via 
the `PySwarmsPlotter` object.
"""
pyswarms_plotter = aplt.PySwarmsPlotter(samples=result.samples)
pyswarms_plotter.cost_history()

"""
__Other Samplers__

Checkout https://pyautofit.readthedocs.io/en/latest/api/api.html for the non-linear searches available in PyAutoFit.
"""
