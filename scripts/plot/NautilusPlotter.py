"""
Plots: NautilusPlotter
======================

This example illustrates how to plot visualization summarizing the results of a nautilus non-linear search using
a `ZeusPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autofit.plot as aplt

"""
First, lets create a result via nautilus by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Nautilus(
    path_prefix="plot",
    name="NautilusPlotter",
    n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `NautilusPlotter` which will allow us to use nautilus's in-built plotting libraries to 
make figures.

The nautilus readthedocs describes fully all of the methods used below 

 - https://nautilus-sampler.readthedocs.io/en/stable/guides/crash_course.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.

Nautilus plotters use `_kwargs` dictionaries to pass visualization settings to matplotlib lib. For example, below,
we:

 - Set the fontsize of the x and y labels by passing `label_kwargs={"fontsize": 16}`.
 - Set the fontsize of the title by passing `title_kwargs={"fontsize": "10"}`.
 
There are other `_kwargs` inputs we pass as None, you should check out the Nautilus docs if you need to customize your
figure.
"""
search_plotter = aplt.NautilusPlotter(
    samples=samples,
    #   output=aplt.Output(path=".", filename="cornerplot", format="png"),
)

"""
The `cornerplot` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
search_plotter.cornerplot(
    panelsize=3.5,
    yticksize=16,
    xticksize=16,
    bins=20,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=True,
    levels=(0.68, 0.95),
    labelpad=0.02,
    range=np.ones(model.total_free_parameters) * 0.999,
    label_kwargs={"fontsize": 24},
)

"""
Finish.
"""
