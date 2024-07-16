"""
Feature: Sensitivity Mapping
============================

Bayesian model comparison allows us to take a dataset, fit it with multiple models and use the Bayesian evidence to
quantify which model objectively gives the best-fit following the principles of Occam's Razor.

However, a complex model may not be favoured by model comparison not because it is the 'wrong' model, but simply
because the dataset being fitted is not of a sufficient quality for the more complex model to be favoured. Sensitivity
mapping addresses what quality of data would be needed for the more complex model to be favoured.

In order to do this, sensitivity mapping involves us writing a function that uses the model(s) to simulate a dataset.
We then use this function to simulate many datasets, for different models, and fit each dataset to quantify
how much the change in the model led to a measurable change in the data. This is called computing the sensitivity.

How we compute the sensitivity is chosen by us, the user. In this example, we will perform multiple model-fits
with a nested sampling search, and therefore perform Bayesian model comparison to compute the sensitivity. This allows 
us to infer how much of a Bayesian evidence increase we should expect for datasets of varying quality and / or models 
with different parameters.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects you have seen elsewhere in the workspace.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

"""
__Data__

Load data of a 1D Gaussian from a .json file in the directory 
`autofit_workspace/dataset/gaussian_x1_with_feature`.

This 1D data includes a small feature to the right of the central `Gaussian`. This feature is a second `Gaussian` 
centred on pixel 70. 
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_with_feature")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Lets plot the data. 

The feature on pixel 70 is clearly visible.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.title("1D Gaussian Data With Feature at pixel 70.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Analysis__

Create the analysis which fits the model to the data.

It fits the data as the sum of the two `Gaussian`'s in the model.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Model Comparison__

Before performing sensitivity mapping, we will quickly perform Bayesian model comparison on this data to get a sense 
for whether the `Gaussian` feature is detectable and how much the Bayesian evidence increases when it is included in
the model.

We therefore fit the data using two models, one where the model is a single `Gaussian` and one where it is 
two `Gaussians`. 

To avoid slow model-fitting and more clearly prounce the results of model comparison, we restrict the centre of 
the`gaussian_feature` to its true centre of 70 and sigma value of 0.5.
"""
model = af.Collection(gaussian_main=af.ex.Gaussian)

search = af.DynestyStatic(
    path_prefix=path.join("features", "sensitivity_mapping", "single_gaussian"),
    nlive=100,
)

result_single = search.fit(model=model, analysis=analysis)

model = af.Collection(gaussian_main=af.ex.Gaussian, gaussian_feature=af.ex.Gaussian)
model.gaussian_feature.centre = 70.0
model.gaussian_feature.sigma = 0.5

search = af.DynestyStatic(
    path_prefix=path.join("features", "sensitivity_mapping", "two_gaussians"), nlive=100
)

result_multiple = search.fit(model=model, analysis=analysis)

"""
We can now print the `log_evidence` of each fit and confirm the model with two `Gaussians` was preferred to the model
with just one `Gaussian`.
"""
print(result_single.samples.log_evidence)
print(result_multiple.samples.log_evidence)

"""
__Sensitivity Mapping__

The model comparison above shows that in this dataset, the `Gaussian` feature was detectable and that it increased the 
Bayesian evidence by ~25. Furthermore, the normalization of this `Gaussian` was ~0.3. 

A lower value of normalization makes the `Gaussian` fainter and harder to detect. We will demonstrate sensitivity 
mapping by answering the following question, at what value of normalization does the `Gaussian` feature become 
undetectable and not provide us with a noticeable increase in Bayesian evidence?

__Base Model__

To begin, we define the `base_model` that we use to perform sensitivity mapping. This model is used to simulate every 
dataset. It is also fitted to every simulated dataset without the extra model component below, to give us the Bayesian
evidence of the every simpler model to compare to the more complex model. 

The `base_model` corresponds to the `gaussian_main` above.
"""
base_model = af.Collection(gaussian_main=af.ex.Gaussian)

"""
__Perturb Model__

We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
sensitivity mapping. Many instances of the `perturb_model` are created and used to simulate the many datasets 
that we fit. However, it is only included in half of the sensitivity mapping models, corresponding to the more complex 
models whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model`.

The `perturb_model` is therefore another `Gaussian` but now corresponds to the `gaussian_feature` above.

By fitting both of these models to every simulated dataset, we will therefore infer the Bayesian evidence of every
model to every dataset. Sensitivity mapping therefore maps out for what values of `normalization` in 
the `gaussian_feature` does the more complex model-fit provide higher values of Bayesian evidence than the simpler
model-fit.
"""
perturb_model = af.Model(af.ex.Gaussian)

"""
__Mapping Grid__

Sensitivity mapping is performed over a large grid of model parameters. To make this demonstration quick and clear we 
are going to fix the `centre` and `sigma` values to the true values of the `gaussian_feature`. We will also iterate 
over just two `normalization` values corresponding to 0.01 and 100.0, which will exhaggerate the difference in
sensitivity between the models at these two values.
"""
perturb_model.centre = 70.0
perturb_model.sigma = 0.5
perturb_model.normalization = af.UniformPrior(lower_limit=0.01, upper_limit=100.0)

"""
__Simulation Instance__

We are performing sensitivity mapping to determine how bright the `gaussian_feature` needs to be in order to be 
detectable. However, every simulated dataset must include the `main_gaussian`, as its presence in the data will effect
the detectability of the `gaussian_feature`.

We can pass the `main_gaussian` into the sensitivity mapping as the `simulation_instance`, meaning that it will be used 
in the simulation of every dataset. For this example we use the inferred `main_gaussian` from one of the model-fits
performed above.
"""
simulation_instance = result_single.instance

"""
__Simulate Function Class__

We are about to write a `simulate_cls` that simulates examples of 1D `Gaussian` datasets that are fitted to
perform sensitivity mapping.

To pass each simulated data through **PyAutoFit**'s sensitivity mapping tools, the function must return a single 
Python object. We therefore define a `Dataset` class that combines the `data` and `noise_map` that are to be 
output by this `simulate_cls`.

It is also convenient to define a `Analysis` class, which behaves analogously to the `Analysis` class used in
PyAutoFit to fit a model to data. In this example it makes it easy to define how we fit each simulated dataset.
"""


class Dataset:
    def __init__(self, data, noise_map):
        self.data = data
        self.noise_map = noise_map


class Analysis(af.ex.Analysis):
    def __init__(self, dataset):
        super().__init__(data=dataset.data, noise_map=dataset.noise_map)


"""
We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
simulate a dataset which is subsequently fitted. 

Additional attributes required to simulate the data can be passed to the `__init__` method, and the simulation is 
performed in the `__call__` method.

Note that when this dataset is simulated, the quantity `instance.perturb` is used in `__call__`.
This is an instance of the `gaussian_feature`, and it is different every time the `simulate_cls` is called. 

In this example, this `instance.perturb` corresponds to two different `gaussian_feature` with values of
`normalization` of 0.01 and 100.0, such that our simulated datasets correspond to a very faint and very bright gaussian 
features .
"""


class Simulate:
    def __init__(self):
        """
        Class used to simulate every dataset used for sensitivity mapping.

        This `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        simulated in the `__call__` simulate_function below.

        In this example we leave it empty as our `simulate_function` does not require any additional information.
        """
        pass

    def __call__(self, instance, simulate_path):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each dataset fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Use the input sensitivity `instance` to simulate the data with the small Gaussian feature.

        2) Output information about the simulation to hard-disk.

        3) Return the data for the sensitivity mapper to fit.

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the Gaussian feature parameters are varied to perform sensitivity.
            The Gaussian feature in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a Gaussian, which i input into the fits of the sensitivity mapper.
        """

        """
        Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated 
        using and thus defining the number of data-points in our data.
        """
        pixels = 100
        xvalues = np.arange(pixels)

        """
        Evaluate the `Gaussian` and Exponential model instances at every xvalues to create their model profile 
        and sum them together to create the overall model profile.

        This print statement will show that, when you run `Sensitivity` below the values of the perturbation 
        use fixed  values of `centre=70` and `sigma=0.5`, whereas the normalization varies over the `number_of_steps` 
        based on its prior.
        """

        print(instance.perturb.centre)
        print(instance.perturb.normalization)
        print(instance.perturb.sigma)

        model_line = instance.gaussian_main.model_data_from(
            xvalues=xvalues
        ) + instance.perturb.model_data_from(xvalues=xvalues)

        """
        Determine the noise (at a specified signal to noise level) in every pixel of our model profile.
        """
        signal_to_noise_ratio = 25.0
        noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

        """
        Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio 
        to compute noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
        """
        data = model_line + noise
        noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

        return Dataset(data=data, noise_map=noise_map)


"""
__Base Fit__

We have defined a `Simulate` class that will be used to simulate every dataset simulated by the sensitivity mapper.
Each simulated dataset will have a unique set of parameters for the `gaussian_feature` (e.g. due to different values of
`perturb_model`.

We will fit each simulated dataset using the `base_model`, which quantifies whether not including the Gaussian feature
in the model changes the goodness-of-fit and therefore indicates if we are sensitive to the Gaussian feature.

We now write a `BaseFit` class, defining how the `base_model` is fitted to each simulated dataset and 
the goodness-of-fit used to quantify whether the model fits the data well. As above, the `__init__` method can be
extended with new inputs to control how the model is fitted and the `__call__` method performs the fit.

In this example, we use a full non-linear search to fit the `base_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `base_model` fit
to the simulated data.
"""


class BaseFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the extra Gaussian feature, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the normalization of the Gaussian
        feature).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.DynestyStatic(
            paths=paths.for_sub_analysis(analysis_name="[base]"),
            nlive=50,
            iterations_per_update=50000,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


"""
__Perturb Fit__

We now define a `PerturbFit` class, which defines how the `perturb_model` is fitted to each simulated dataset. This
behaves analogously to the `BaseFit` class above, but now fits the `perturb_model` to the simulated data (as
opposed to the `base_model`).

Again, in this example we use a full non-linear search to fit the `perturb_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `perturb_model` fit
to the simulated data.
"""


class PerturbFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the extra Gaussian feature, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the normalization of the Gaussian
        feature).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.DynestyStatic(
            paths=paths.for_sub_analysis(analysis_name="[perturbed]"),
            nlive=50,
            iterations_per_update=50000,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


"""
We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
object below are:

- `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
example it contains an instance of the `gaussian_main` model component.

- `base_model`: This is the simpler model that is fitted to every simulated dataset, which in this example is composed 
of a single `Gaussian` called the `gaussian_main`.

- `perturb_model`: This is the extra model component that has two roles: (i) based on the sensitivity grid parameters
it is added to the `simulation_instance` to simulate each dataset ; (ii) it is added to the`base_model` and fitted to 
every simulated dataset (in this example every `simulation_instance` and `perturb_model` there has two `Gaussians` 
called the `gaussian_main` and `gaussian_feature`).

- `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model` 
to simulate many datasets which are fitted with the `base_model` and `base_model` + `perturb_model`.

- `base_fit_cls`: This is the function that fits the `base_model` to every simulated dataset and returns the
goodness-of-fit of the model to the data.

- `perturb_fit_cls`: This is the function that fits the `base_model` + `perturb_model` to every simulated dataset and
returns the goodness-of-fit of the model to the data.

- `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this 
example, normalization has a `LogUniformPrior` with lower limit 1e-4 and upper limit 1e2, therefore the `number_of_steps` 
of 2 wills imulate and fit just 2 datasets where the intensities between 1e-4 and 1e2.

- `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
if set above 1.
"""
paths = af.DirectoryPaths(
    path_prefix=path.join("features"),
    name="sensitivity_mapping",
)

sensitivity = af.Sensitivity(
    paths=paths,
    simulation_instance=simulation_instance,
    base_model=base_model,
    perturb_model=perturb_model,
    simulate_cls=Simulate(),
    base_fit_cls=BaseFit(analysis_cls=Analysis),
    perturb_fit_cls=PerturbFit(analysis_cls=Analysis),
    number_of_steps=2,
    number_of_cores=2,
)
sensitivity_result = sensitivity.run()

"""
__Results__

You should now look at the results of the sensitivity mapping in the folder `output/features/sensitivity_mapping`. 

You will note the following 4 model-fits have been performed:

 - The `base_model` is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=0.01` are used.

 - The `base_model` + `perturb_model`  is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=0.01` are used.

 - The `base_model` is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=100.0` are used.

 - The `base_model` + `perturb_model`  is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=100.0` are used.

The fit produced a `sensitivity_result`. 

We are still developing the `SensitivityResult` class to provide a data structure that better streamlines the analysis
of results. If you intend to use sensitivity mapping, the best way to interpret the resutls is currently via
**PyAutoFit**'s database and `Aggregator` tools. 
"""
print(sensitivity_result.samples)
print(sensitivity_result.log_evidences_base)

"""
Finish.
"""
