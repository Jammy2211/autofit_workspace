"""
Plots: EmceePlotter
===================

This example illustrates how to plot visualization summarizing the results of a emcee non-linear search using
a `EmceePlotter`.
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
First, lets create a result via emcee by repeating the simple model-fit that is performed in 
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

search = af.Emcee(
    path_prefix=path.join("plot"), name="EmceePlotter", nwalkers=100, nsteps=500
)

result = search.fit(model=model, analysis=analysis)

"""
We now pass the samples to a `EmceePlotter` which will allow us to use emcee's in-built plotting libraries to 
make figures.

The emcee readthedocs describes fully all of the methods used below 

 - https://emcee.readthedocs.io/en/stable/user/sampler/
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
samples = result.samples

search_plotter = aplt.EmceePlotter(samples=samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
search_plotter.corner(
    bins=20,
    range=None,
    color="k",
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
)

"""
The `trajectories` method shows the likelihood of every parameter as a function of parameter value, colored by every
individual walker.
"""
search_plotter.trajectories()

"""
The `likelihood_series` method shows the likelihood as a function of step number, colored by every individual walker.
"""
search_plotter.time_series()

"""
The `time_series` method shows the likelihood of every parameter as a function of step number, colored by every
individual walker.
"""
search_plotter.likelihood_series()

"""
Finish.
"""
