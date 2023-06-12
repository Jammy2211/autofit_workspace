"""
Plots: PySwarmsPlotter
======================

This example illustrates how to plot visualization summarizing the results of a pyswarms non-linear search using
a `ZeusPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autofit.plot as aplt

"""
First, lets create a result via pyswarms by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.PySwarmsGlobal(
    path_prefix=path.join("plot"), name="PySwarmsPlotter", n_particles=50, iters=10
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `PySwarmsPlotter` which will allow us to use pyswarms's in-built plotting libraries to 
make figures.

The pyswarms readthedocs describes fully all of the methods used below 

 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.plotters.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
pyswarms_plotter = aplt.PySwarmsPlotter(samples=samples)

"""
The `contour` method shows a 2D projection of the particle trajectories.
"""
pyswarms_plotter.contour(
    canvas=None,
    title="trajectories",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
)


"""
The `cost history` shows in 1D the evolution of each parameters estimated highest likelihood.
"""
pyswarms_plotter.cost_history(ax=None, title="Cost History", designer=None)

"""
The `trajectories` method shows the likelihood of every parameter as a function of parameter value.
"""
pyswarms_plotter.trajectories()

"""
The `time_series` method shows the likelihood of every parameter as a function of step number.
"""
pyswarms_plotter.time_series()

"""
Finish.
"""
