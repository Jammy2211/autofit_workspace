"""
Plots: DynestyPlotter
======================

This example illustrates how to plot visualization summarizing the results of a nautilus non-linear search using
a `MCMCPlotter`.
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
    name="NestPlotter",
    n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
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

We now pass the samples to a `NestPlotter` which will allow us to use nautilus's in-built plotting libraries to 
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
plotter = aplt.NestPlotter(
    samples=samples,
)

"""
The `corner_anesthetic` method produces a triangle of 1D and 2D PDF's of every parameter using the library `anesthetic`.
"""
plotter.corner_anesthetic()

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
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
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
__Plots__

Nautilus example plots are not shown explicitly below, so checkout their docs for examples!
"""
