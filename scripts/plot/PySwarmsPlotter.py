"""
Plots: MLEPlotter
======================

This example illustrates how to plot visualization summarizing the results of a pyswarms non-linear search using
a `MLEPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
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
    path_prefix=path.join("plot"), name="MLEPlotter", n_particles=50, iters=10
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__Plotting__

We now pass the samples to a `MLEPlotter` which will allow us to use pyswarms's in-built plotting libraries to 
make figures.

The pyswarms readthedocs describes fully all of the methods used below 

 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.plotters.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.MLEPlotter(samples=samples)

"""
__Search Specific Visualization__

PySwarms has bespoke in-built visualization tools that can be used to plot its results.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
The `contour` method shows a 2D projection of the particle trajectories.
"""
from pyswarms.utils import plotters

plotters.plot_contour(
    pos_history=search_internal,
    canvas=None,
    title="Trajectories",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
)
plt.show()

plotters.plot_cost_history(
    cost_history=search_internal.cost_history,
    ax=None,
    title="Cost History",
    designer=None,
)
plt.show()

"""
Finish.
"""
