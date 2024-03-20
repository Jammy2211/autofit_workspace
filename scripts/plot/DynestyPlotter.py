"""
Plots: NestPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a dynesty non-linear search using
a `NestPlotter`.
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
First, lets create a result via dynesty by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(path_prefix="plot", name="NestPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
__Plotting__

We now pass the samples to a `NestPlotter` which will allow us to use dynesty's in-built plotting libraries to 
make figures.

The dynesty readthedocs describes fully all of the methods used below 

 - https://dynesty.readthedocs.io/en/latest/quickstart.html
 - https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.plotting
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.

Dynesty plotters use `_kwargs` dictionaries to pass visualization settings to matplotlib lib. For example, below,
we:

 - Set the fontsize of the x and y labels by passing `label_kwargs={"fontsize": 16}`.
 - Set the fontsize of the title by passing `title_kwargs={"fontsize": "10"}`.
 
There are other `_kwargs` inputs we pass as None, you should check out the Dynesty docs if you need to customize your
figure.
"""
plotter = aplt.NestPlotter(samples=samples)

"""
The `corner_anesthetic` method produces a triangle of 1D and 2D PDF's of every parameter using the library `anesthetic`.
"""
plotter.corner_anesthetic()

"""
The `corner_cornerpy` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
    dims=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=0.02,
    quantiles_2d=None,
    hist_kwargs=None,
    hist2d_kwargs=None,
    label_kwargs={"fontsize": 16},
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": "10"},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
)

"""
Finish.
"""
