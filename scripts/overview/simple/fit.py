"""
__Example: Fit__

In this example, we'll fit 1D data of a `Gaussian` profile with a 1D `Gaussian` model using the non-linear searches
emcee, Dynesty and PySwarms.

If you haven't already, you should checkout the files `example/model.py` and `example/analysis.py` to see how we have
provided PyAutoFit with the necessary information on our model, data and log likelihood function.
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

Load data of a 1D Gaussian from a .json file in the directory 
`autofit_workspace/dataset//gaussian_x1`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
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

We now need to create our model of a 1D Gaussian, which is the model we will fit to the dataset above.

A model component is written as a Python class in *PyAutoFit** using the following format:

 - The name of the class is the name of the model component, in this case, “Gaussian”.

 - The input arguments of the constructor are the parameters of the mode (here centre, normalization and sigma).

 - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
 multi-valued tuple.

Below, we define a 1D Gaussian model component, which is used throughout the **PyAutoFit** workspace to perform
example model fits.
"""


class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """
        Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = xvalues - self.centre

        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


"""
We can convert a Python class written using the format above to a model-component via the `af.Model()` object.

This inspects the `__init__` constructor defined above, to determine that this model-component has 3 free
parameters (`centre`, `normalization`, `sigma`).
"""
model = af.Model(Gaussian)

"""
We can check it has 3 free parameters via the `prior_count` attribute.
"""
print(f"Model Prior Count = {model.prior_count}")

"""
By default, the priors associated with every parameter of the model are loaded from a user-defined configuration file.

For the `Gaussian` above, these priros are contained in the file `autofit_workspace/config/priors/model.json`, which
you can checkout now to see the priors.

We can print the priors via the model's `info` attribute.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
We can overwrite priors before fitting the model to data:
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
The `info` attribute shows the priors on parameters have been overwritten by those specified above:
"""
print(model.info)

"""
__Analysis__

We now set up our Analysis using an `Analysis` object, where:

 - The `__init__` constructor of the `Analysis` object contains the data, noise-map and any other quantities necessary
 to fit the model to the data.

 - The `log_likelihood_function` defines how given an instance of our model (a Gaussian) we fit the data and return a 
 log likelihood value.
 
 - The `visualize` function outputs visuals to hard-disk during and at the end of the model-fit.
   
Read the comments and docstrings of the `Analysis` object below in detail for more insights into how this object
works.
"""


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        """
        In this example the `Analysis` object only contains the data and noise-map. It can be easily extended,
        for more complex data-sets and model fitting problems.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D Gaussian) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    """
    In the log_likelihood_function function below, `instance` is an instance of our model, which in this example is
    an instance of the `Gaussian` class in `model.py`. The parameters of the `Gaussian` are set via the non-linear
    search. This gives us the instance of our model we need to fit our data!
    """

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of a `Gaussian` to the dataset, using a model instance of the Gaussian.

        This function will be replaced with calculations specific to your model-fitting problem.

        Parameters
        ----------
        instance
        An instance of the `Gaussian` model, where the parameters have been set via the non-linear search of the
        model-fit.

        Returns
        -------
        fit
            The log likelihood value indicating how well this model fit the dataset.
        """

        """"
        The `instance` that comes into this method is an instance of the `Gaussian` class. To convince yourself of 
        this, go ahead and uncomment the lines below and run the non-linear search.
        """
        # print("Gaussian Instance:")
        # print("Centre = ", instance.centre)
        # print("Normalization = ", instance.normalization)
        # print("Sigma = ", instance.sigma)

        """
        Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
        """
        xvalues = np.arange(self.data.shape[0])

        """
        Use these xvalues to create model data of our Gaussian.
        """
        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        """
        Fit the model gaussian line data to the observed data, computing the residuals and chi-squared.
        """
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search and is used to output
        images indicating the quality of the fit so far..

        The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
        so far and it can be used to provide on-the-fly images showing how the model-fit is going.

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data, color="r")
        plt.title("Model fit to 1D Gaussian dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.clf()


"""
Now we have defined out `Analysis` object we create an instance of it to perform our model-fit.
"""
analysis = Analysis(data=data, noise_map=noise_map)

"""
#############################
###### NESTED SAMPLING ######
#############################

We now set up our non-linear search. We'll first fit the data with the nested sampling algorithm
Dynesty. Below, we manually specify all of the Dynesty settings, however if we omitted them the default values
found in the config file `config/non_linear/Dynesty.yaml` would be used.

For a full description of Dynesty checkout its Github and documentation webpages:

https://github.com/joshspeagle/dynesty
https://dynesty.readthedocs.io/en/latest/index.html
"""
search = af.DynestyStatic(
    nlive=100,
    bound="multi",
    sample="auto",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    number_of_cores=1,
)

"""
To perform the fit with Dynesty, we pass it our model and analysis and we`re good to go!

Checkout the folder `autofit_workspace/output/DynestyStatic`, where the `NonLinearSearch` results, visualization and
information can be found.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. 

The `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
We can inspect the result's maximum likelihood instance.
"""
print(result.max_log_likelihood_instance)

print("\n Model-fit Max Log-likelihood Parameter Estimates: \n")
print("Centre = ", result.max_log_likelihood_instance.centre)
print("Normalization = ", result.max_log_likelihood_instance.normalization)
print("Sigma = ", result.max_log_likelihood_instance.sigma)

"""
We can use the result's maximum likelihood instance to compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Dynesty model fit to 1D Gaussian dataset.")
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
__Output to Hard Disk__

The non-linear search `dynesty` above did not output results to hard-disk, which for quick model-fits and
experimenting with different models is desirable.

For many problems it is preferable for all results to be written to hard-disk. The benefits of doing this include:

- Inspecting results in an ordered directory structure can be more efficient than using a Jupyter Notebook.
- Results can be output on-the-fly, to check that a fit is progressing as expected mid way through.
- An unfinished run can be resumed where it was terminated.
- Additional information about a fit (e.g. visualization) can be output.
- On high performance computers which use a batch system, this is the only way to transfer results.

Any model-fit performed by **PyAutoFit** can be saved to hard-disk, by simply giving the non-linear search a 
`name`. A `path_prefix` can optionally be input to customize the output directory.
"""
search = af.DynestyStatic(
    path_prefix=path.join("overview", "simple"),
    name="DynestyStatic",
    nlive=100,
    bound="multi",
    sample="auto",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    walks=25,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
In the `autofit_workspace/output` directory you should find a folder containing the results of the model-fit
for inspection, including text files containing the `model.info`, `results.info` and other information.

The results are in a folder which is a collection of random characters. This is the 'unique_identifier' of
the model-fit. This identifier is generated based on the model fitted and search used, such that an identical
combination of model and search generates the same identifier. 

This ensures that rerunning an identical fit will use the existing results to resume the model-fit. In contrast, if
you change the model or search, a new unique identifier will be generated, ensuring that the model-fit results are
output into a separate folder.

We discuss in more detail how to use a results object in the files `autofit_workspace/example/simple/result.py`.

##################
###### MCMC ######
##################

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
    path_prefix=path.join("overview", "simple"),
    name="Emcee",
    nwalkers=30,
    nsteps=1000,
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
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("Emcee model fit to 1D Gaussian dataset.")
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
search = af.PySwarmsGlobal(
    path_prefix=path.join("overview", "simple"),
    name="PySwarmsGlobal",
    n_particles=50,
    iters=100,
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
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.plot(xvalues, model_data, color="r")
plt.title("PySwarms model fit to 1D Gaussian dataset.")
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
