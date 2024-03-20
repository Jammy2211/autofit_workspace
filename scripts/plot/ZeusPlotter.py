"""
Plots: MCMCPlotter
==================

This example illustrates how to plot visualization summarizing the results of a zeus non-linear search using
a `MCMCPlotter`.
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
First, lets create a result via zeus by repeating the simple model-fit that is performed in 
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

search = af.Zeus(
    path_prefix=path.join("plot"), name="MCMCPlotter", nwalkers=100, nsteps=10000
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `MCMCPlotter` which will allow us to use dynesty's in-built plotting libraries to 
make figures.

The zeus readthedocs describes fully all of the methods used below 

 - https://zeus-mcmc.readthedocs.io/en/latest/api/plotting.html#cornerplot
 - https://zeus-mcmc.readthedocs.io/en/latest/notebooks/normal_distribution.html
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
zeus_plotter = aplt.MCMCPlotter(samples=samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
zeus_plotter.corner(
    weight_list=None,
    levels=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    truth=None,
    color=None,
    alpha=0.5,
    linewidth=1.5,
    fill=True,
    fontsize=10,
    show_titles=True,
    title_fmt=".2f",
    title_fontsize=12,
    cut=3,
    fig=None,
    size=(10, 10),
)

"""
Finish.
"""
