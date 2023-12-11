"""
Feature: Interpolate
====================

On may have multiple datasets where it is anticipated that one or more parameters vary smoothly across the datasets.

For example, the datasets may be taken at different times, where the signal in the data and therefore model parameters
one infers vary smoothly as a function of time. Alternaitvely, the datasets may be taken at different wavelengths,
with the signal varying smoothly as a function of wavelength.

In any of these cases, it could be desireable to fit the datasets one-by-one, and then interpolate the results in order
to determine the most likely model parameters at any point in time (or at any wavelength).

This example illustrates interpolate functionality in **PyAutoFit** using the example of fitting 3 noisy 1D Gaussians,
where these data are assumed to have been taken at 3 different times. The `centre` of each `Gaussian` varies smoothly
over time. The interpolation is therefore used to estimate the `centre` of each `Gaussian` at any time between the
three times the data was taken.

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

import numpy as np
import matplotlib.pyplot as plt
from os import path

import autofit as af

"""
__Dataset__

We load 3 noisy 1D Gaussian datasets taken at 3 different times, where the `centre` of each `Gaussian` varies smoothly 
over time.

The datasets are taken at 3 times, t=0, t=1 and t=2, which defines the name of the folder we load the data from.

We load each data and noise map and store them in lists, so we can plot them next.
"""
total_datasets = 3

data_list = []
noise_map_list = []
time_list = []

for time in range(3):
    dataset_name = f"time_{time}"
    
    dataset_prefix_path = path.join("dataset", "example_1d", "gaussian_x1_variable")

    dataset_path = path.join(
        dataset_prefix_path, dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)
    time_list.append(time)

"""
Now lets plot the data, including their error bars.

Visual comparison of the datasets shows that the `centre` of each `Gaussian` varies smoothly over time, with it moving
from pixel 40 at t=0 to pixel 60 at t=2.
"""
for time in range(3):
    
    xvalues = range(data_list[time].shape[0])

    plt.errorbar(
        x=xvalues,
        y=data_list[time],
        yerr=noise_map_list[time],
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Data #1.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()


"""
We now fit each of the 3 datasets.

The fits are performed in a for loop, with the docstrings inside the loop explaining the code.

The interpolate at the end of the fits uses the maximum log likelihood model of each fit, which we therefore store 
in a list.
"""
ml_instances_list = []

for data, noise_map, time in zip(data_list, noise_map_list, time_list):

    """
    __Analysis__
    
    For each dataset we create an `Analysis` class, which includes the `log_likelihood_function` we fit the data with.
    """
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    """
    __Time__
    
    The model composed below has an input not seen in other scripts, the parameter `time`.
    
    This is the time that the simulated data was acquired, and is not a free parameter in the fit. 
    
    For interpolation it plays a crucial role, as the model is interpolated to the time of every dataset as input
    into the model below. If the `time` input were missing, interpolation could not be performed.
    
    Over the iterations of the for loop, the `time` input will therefore be the values 0.0, 1.0 and 2.0.

    __Model__
    
    We now compose our model, which is a single `Gaussian`.
    
    The `centre` of the `Gaussian` is a free parameter with a `UniformPrior` that ranges between 0.0 and 100.0. 
    
    We expect the inferred `centre` inferred from the fit to each dataset to vary smoothly as a function of time.
    """
    model = af.Collection(
        gaussian=af.Model(af.ex.Gaussian),
        time=time
    )

    """
    __Search__
    
    The model is fitted to the data using the nested sampling algorithm 
    Dynesty (https://johannesbuchner.github.io/UltraNest/readme.html).
    """
    search = af.DynestyStatic(
        path_prefix=path.join("interpolate"),
        name=f"time_{time}",
        nlive=100,
    )

    """
    __Model-Fit__
    
    We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
    search to find which models fit the data with the highest likelihood.
    """
    result = search.fit(model=model, analysis=analysis)

    """
    __Instances__
    
    Interpolation uses the maximum log likelihood model of each fit to build an interpolation model of the model as a
    function of time. 
    
    We therefore store the maximum log likelihood model of every fit in a list, which is used below.
    """
    ml_instances_list.append(result.instance)


"""
__Interpolation__

Now all fits are complete, we use the `ml_instances_list` to build an interpolation model of the model as a function 
of time.

This is performed using the `LinearInterpolator` object, which interpolates the model parameters as a function of
time linearly between the values computed by the model-fits above.

More advanced interpolation schemes are available and described in the `interpolation.py` example.
"""
interpolator = af.LinearInterpolator(instances=ml_instances_list)

"""
The model can be interpolated to any time, for example time=1.5.

This returns a new `instance` of the model, as an instance of the `Gaussian` object, where the parameters are computed 
by interpolating between the values computed above.
"""
instance = interpolator[interpolator.time == 1.5]

"""
The `centre` of the `Gaussian` at time 1.5 is between the value inferred for the first and second fits taken
at times 1.0 and 2.0.

This is a `centre` close to a value of 55.0.
"""
print(f"Gaussian centre of fit 1 (t = 1): {ml_instances_list[0].gaussian.centre}")
print(f"Gaussian centre of fit 2 (t = 2): {ml_instances_list[1].gaussian.centre}")

print(f"Gaussian centre interpolated at t = 1.5 {instance.gaussian.centre}")


"""
__Serialization__

The interpolator and model can be serialized to a .json file using **PyAutoConf**'s dedicated serialization methods. 

This means an interpolator can easily be loaded into other scripts.
"""
from autoconf.dictable import output_to_json, from_json

json_file = path.join(dataset_prefix_path, "interpolator.json")

output_to_json(obj=interpolator, file_path=json_file)

interpolator = from_json(file_path=json_file)

"""
__Database__

It may be inconvenient to fit all the models in a single Python scfript, especially if the model-fits take a long time
and you are fitting many datasets.

PyAutoFit's database tools allow you to store the results of model-fits from hard-disk. 

Below, we use these tools to load the results of the fit above, set up the interpolator and perform the interpolation.

If you are not familiar with the database API, you should checkout the `cookbook/database.ipynb` example.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "interpolate"), completed_only=False
)

ml_instances_list = [samps.max_log_likelihood() for samps in agg.values("samples")]

interpolator = af.LinearInterpolator(instances=ml_instances_list)

instance = interpolator[interpolator.time == 1.5]

print(f"Gaussian centre of fit 1 (t = 1): {ml_instances_list[0].gaussian.centre}")
print(f"Gaussian centre of fit 2 (t = 2): {ml_instances_list[1].gaussian.centre}")

print(f"Gaussian centre interpolated at t = 1.5 {instance.gaussian.centre}")

