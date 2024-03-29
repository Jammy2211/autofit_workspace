"""
Plots: DynestyPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a dynesty non-linear search using
a `DynestyPlotter`.
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

search = af.DynestyStatic(path_prefix="plot", name="DynestyPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `DynestyPlotter` which will allow us to use dynesty's in-built plotting libraries to 
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
search_plotter = aplt.DynestyPlotter(samples=samples)

"""
The `cornerplot` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
search_plotter.cornerplot(
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
The `runplot` method shows how the estimates of the log evidence and other quantities progress as a function of
iteration number during the dynesty model-fit.
"""
search_plotter.runplot(
    span=None,
    logplot=False,
    kde=True,
    nkde=1000,
    color="blue",
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    lnz_error=True,
    lnz_truth=None,
    truth_color="red",
    truth_kwargs=None,
    max_x_ticks=8,
    max_y_ticks=3,
    use_math_text=True,
    mark_final_live=True,
    fig=None,
)

"""
The `traceplot` method shows how the live points of each parameter converged alongside their PDF.
"""
search_plotter.traceplot(
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    smooth=0.02,
    thin=1,
    dims=None,
    post_color="blue",
    post_kwargs=None,
    kde=True,
    nkde=1000,
    trace_cmap="plasma",
    trace_color=None,
    trace_kwargs=None,
    connect=False,
    connect_highlight=10,
    connect_color="red",
    connect_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    label_kwargs={"fontsize": 16},
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": "10"},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    verbose=False,
    fig=None,
)


"""
The `cornerpoints` method produces a triangle of 1D and 2D plots of the weight points of every parameter in the model 
fit.
"""
search_plotter.cornerpoints(
    dims=None,
    thin=1,
    span=None,
    cmap="plasma",
    color=None,
    kde=True,
    nkde=1000,
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    use_math_text=False,
    fig=None,
)

"""
The `boundplot` method produces a plot of the bounding distribution used to draw a live point at a given iteration `it`
of the sample or of a dead point `idx`.
"""
search_plotter.boundplot(
    dims=(2, 2),
    it=100,
    idx=None,
    prior_transform=None,
    periodic=None,
    reflective=None,
    ndraws=5000,
    color="gray",
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    max_n_ticks=5,
    use_math_text=False,
    show_live=False,
    live_color="darkviolet",
    live_kwargs=None,
    span=None,
    fig=None,
)

"""
The `cornerbound` method produces the bounding distribution used to draw points at an input iteration `it` or used to
specify a dead point via `idx`.
"""
search_plotter.cornerbound(
    it=100,
    idx=None,
    dims=None,
    prior_transform=None,
    periodic=None,
    reflective=None,
    ndraws=5000,
    color="gray",
    plot_kwargs=None,
    label_kwargs={"fontsize": 16},
    max_n_ticks=5,
    use_math_text=False,
    show_live=False,
    live_color="darkviolet",
    live_kwargs=None,
    span=None,
    fig=None,
)

"""
Finish.
"""
