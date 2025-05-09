"""
Plots: DynestyPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a dynesty non-linear search using
a `NestPlotter`.
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
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

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

All plots use dynesty's inbuilt plotting library and the model.
"""
from dynesty import plotting as dyplot

model = result.model

"""
The boundplot plots the bounding distribution used to propose either (1) live points at a given iteration or (2) a 
specific dead point during the course of a run, projected onto the two dimensions specified by `dims`.
"""
dyplot.boundplot(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    dims=(2, 2),
    it=-1,  # The iteration number to make the plot.
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

plt.show()
plt.close()

"""
The cornerbound plots the bounding distribution used to propose either (1) live points at a given iteration or (2) a 
specific dead point during the course of a run, projected onto all pairs of dimensions.
"""
dyplot.cornerbound(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    it=-1,  # The iteration number to make the plot.
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

plt.show()
plt.close()

"""
The cornerplot plots a corner plot of the 1-D and 2-D marginalized posteriors.
"""

try:
    dyplot.cornerplot(
        results=search_internal.results,
        labels=model.parameter_labels_with_superscripts_latex,
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

    plt.show()
    plt.close()

except ValueError:
    pass

"""
The cornerpoints plots a (sub-)corner plot of (weighted) samples.
"""
dyplot.cornerpoints(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
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

plt.show()
plt.close()


"""
The runplot plots live points, ln(likelihood), ln(weight), and ln(evidence) as a function of ln(prior volume).
"""
dyplot.runplot(
    results=search_internal.results,
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

plt.show()
plt.close()


"""
The traceplot plots traces and marginalized posteriors for each parameter.
"""
try:
    dyplot.traceplot(
        results=search_internal.results,
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

    plt.show()
    plt.close()

except ValueError:
    pass

"""
Finish.
"""
