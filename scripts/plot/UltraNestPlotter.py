"""
Plots: UltraNestPlotter
=======================

This example illustrates how to plot visualization summarizing the results of a ultranest non-linear search using
a `ZeusPlotter`.

Installation
------------

Because UltraNest is an optional library, you will likely have to install it manually via the command:

`pip install ultranest`
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
First, lets create a result via ultranest by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.UltraNest(path_prefix="plot", name="UltraNestPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `UltraNestPlotter` which will allow us to use ultranest's in-built plotting libraries to 
make figures.

The ultranest readthedocs describes fully all of the methods used below 

 - https://johannesbuchner.github.io/UltraNest/readme.html
 - https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.plot
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
ultranest_plotter = aplt.UltraNestPlotter(samples=samples)

"""
The `cornerplot` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
ultranest_plotter.cornerplot()

"""
The `runplot` method shows how the estimates of the log evidence and other quantities progress as a function of
iteration number during the ultranest model-fit.
"""
ultranest_plotter.runplot(
    span=None,
    logplot=False,
    kde=True,
    nkde=1000,
    color="blue",
    plot_kwargs=None,
    label_kwargs=None,
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
ultranest_plotter.traceplot(
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    smooth=0.02,
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
    labels=None,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    verbose=False,
    fig=None,
)

"""
Finish.
"""
