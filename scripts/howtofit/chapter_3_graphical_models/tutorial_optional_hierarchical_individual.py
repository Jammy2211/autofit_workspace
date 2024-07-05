"""
Tutorial Optional: Hierarchical Individual
==========================================

In tutorial 4, we fit a hierarchical model using a graphical model, whereby all datasets are fitted simultaneously
and the hierarchical parameters are fitted for simultaneously with the model parameters of each 1D Gaussian in each
dataset.

This script illustrates how the hierarchical parameters can be estimated using a simpler approach, which fits
each dataset one-by-one and estimates the hierarchical parameters afterwards by fitting the inferred `centres`
with a Gaussian distribution.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and 
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you 
have seen and used elsewhere throughout the workspace.

__Dataset__

For each dataset we now set up the correct path and load it. 

We are loading a new Gaussian dataset, where the Gaussians have different centres which were drawn from a parent
Gaussian distribution with a mean centre value of 50.0 and sigma of 10.0.
"""
total_datasets = 5

dataset_name_list = []
data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__hierarchical", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
By plotting the Gaussians we can just about make out that their centres are not all at pix 50, and are spread out
around it (albeit its difficult to be sure, due to the low signal-to-noise of the data). 
"""
for dataset_name, data in zip(dataset_name_list, data_list):
    af.ex.plot_profile_1d(
        xvalues=np.arange(data.shape[0]),
        profile_1d=data,
        title=dataset_name,
        ylabel="Data Values",
        color="k",
    )

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class, like in the previous tutorial.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)


"""
__Model__

The model we fit to each dataset, which is a simple 1D Gaussian with all 3 parameters free.
"""
gaussian = af.Model(af.ex.Gaussian)

gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

model = af.Collection(gaussian=gaussian)


"""
__Model Fits (one-by-one)__

For every dataset we now create an `Analysis` class using it and use `Dynesty` to fit it with a `Gaussian`.

The `Result` is stored in the list `results`.
"""
result_list = []

for dataset_name, analysis in zip(dataset_name_list, analysis_list):
    """
    Create the `DynestyStatic` non-linear search and use it to fit the data.
    """
    dynesty = af.DynestyStatic(
        name="tutorial_optional_hierarchical_individual",
        unique_tag=dataset_name,
        nlive=200,
        dlogz=1e-4,
        sample="rwalk",
        walks=10,
    )

    result_list.append(dynesty.fit(model=model, analysis=analysis))

"""
__Results__

Checkout the output folder, you should see three new sets of results corresponding to our 3 `Gaussian` datasets.

The `result_list` allows us to plot the median PDF value and 3.0 confidence intervals of the `centre` estimate from
the model-fit to each dataset.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
ue3_instances = [samp.errors_at_upper_sigma(sigma=3.0) for samp in samples_list]
le3_instances = [samp.errors_at_lower_sigma(sigma=3.0) for samp in samples_list]

mp_centres = [instance.gaussian.centre for instance in mp_instances]
ue3_centres = [instance.gaussian.centre for instance in ue3_instances]
le3_centres = [instance.gaussian.centre for instance in le3_instances]

print(f"Median PDF inferred centre values")
print(mp_centres)
print()

"""
__Overall Gaussian Parent Distribution__

Fit the inferred `centre`'s from the fits performed above with a Gaussian distribution, in order to 
estimate the mean and scatter of the Gaussian from which the centres were drawn.

We first extract the inferred median PDF centre values and their 1 sigma errors below, which will be the inputs
to our fit for the parent Gaussian.
"""
ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_centres = [instance.gaussian.centre for instance in ue1_instances]
le1_centres = [instance.gaussian.centre for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres, le1_centres)]

"""
The `Analysis` class below fits a Gaussian distribution to the inferred `centre` values from each of the fits above,
where the inferred error values are used as the errors.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, errors: np.ndarray):
        super().__init__()

        self.data = np.array(data)
        self.errors = np.array(errors)

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Fits a set of 1D data points with a 1D Gaussian distribution, in order to determine from what Gaussian
        distribution the analysis classes `data` were drawn.

        In this example, this function determines from what parent Gaussian disrtribution the inferred centres
        of each 1D Gaussian were drawn.
        """
        log_likelihood_term_1 = np.sum(
            -np.divide(
                (self.data - instance.median) ** 2,
                2 * (instance.scatter**2 + self.errors**2),
            )
        )
        log_likelihood_term_2 = -np.sum(
            0.5 * np.log(instance.scatter**2 + self.errors**2)
        )

        return log_likelihood_term_1 + log_likelihood_term_2


"""
The `ParentGaussian` class is the model-component which used to fit the parent Gaussian to the inferred `centre` values.
"""


class ParentGaussian:
    def __init__(self, median: float = 0.0, scatter: float = 0.01):
        """
        A model component which represents a parent Gaussian distribution, which can be fitted to a 1D set of
        measurments with errors in order to determine the probabilty they were drawn from this Gaussian.

        Parameters
        ----------
        median
            The median value of the parent Gaussian distribution.
        scatter
            The scatter (E.g. the sigma value) of the Gaussian.
        """

        self.median = median
        self.scatter = scatter

    def probability_from_values(self, values: np.ndarray) -> float:
        """
        For a set of 1D values, determine the probability that they were random drawn from this parent Gaussian
        based on its `median` and `scatter` attributes.

        Parameters
        ----------
        values
            A set of 1D values from which we will determine the probability they were drawn from the parent Gaussian.
        """
        values = np.sort(np.array(values))
        transformed_values = np.subtract(values, self.median)

        return np.multiply(
            np.divide(1, self.scatter * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_values, self.scatter))),
        )


"""
__Model__

The `ParentGaussian` is the model component we fit in order to determine the probability the inferred centres were
drawn from the distribution.

This will be fitted via a non-linear search and therefore is created as a model component using `af.Model()` as per 
usual in **PyAutoFit**.
"""
model = af.Model(ParentGaussian)

model.median = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.scatter = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

"""
__Analysis + Search__

We now create the Analysis class above which fits a parent 1D gaussian and create a dynesty search in order to fit
it to the 1D inferred list of `centres`.
"""
analysis = Analysis(data=mp_centres, errors=error_list)
search = af.DynestyStatic(nlive=100)

result = search.fit(model=model, analysis=analysis)

"""
The results of this fit tell us the most probably values for the `median` and `scatter` of the 1D parent Gaussian fit.
"""
samples = result.samples

median = samples.median_pdf().median

u1_error = samples.values_at_upper_sigma(sigma=1.0).median
l1_error = samples.values_at_lower_sigma(sigma=1.0).median

u3_error = samples.values_at_upper_sigma(sigma=3.0).median
l3_error = samples.values_at_lower_sigma(sigma=3.0).median

print(
    f"Inferred value of the hierarchical median via simple fit to {total_datasets} datasets: \n "
)
print(f"{median} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{median} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()

scatter = samples.median_pdf().scatter

u1_error = samples.values_at_upper_sigma(sigma=1.0).scatter
l1_error = samples.values_at_lower_sigma(sigma=1.0).scatter

u3_error = samples.values_at_upper_sigma(sigma=3.0).scatter
l3_error = samples.values_at_lower_sigma(sigma=3.0).scatter

print(
    f"Inferred value of the hierarchical scatter via simple fit to {total_datasets} datasets: \n "
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()

"""
We can compare these values to those inferred in `tutorial_4_hierarchical_model`, which fits all datasets and the
hierarchical values of the parent Gaussian simultaneously.,
 
The errors for the fit performed in this tutorial are much larger. This is because of how in a graphical model
the "datasets talk to one another", which is described fully in that tutorials subsection "Benefits of Graphical Model".
"""
