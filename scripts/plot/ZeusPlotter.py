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

import matplotlib.pyplot as plt
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
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__Plotting__

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
plotter = aplt.MCMCPlotter(samples=samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
plotter.corner_cornerpy(
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
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
The method below shows a 2D projection of the walker trajectories.
"""
fig, axes = plt.subplots(result.model.prior_count, figsize=(10, 7))

for i in range(result.model.prior_count):
    for walker_index in range(search_internal.get_log_prob().shape[1]):
        ax = axes[i]
        ax.plot(
            search_internal.get_chain()[:, walker_index, i],
            search_internal.get_log_prob()[:, walker_index],
            alpha=0.3,
        )

    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel(result.model.parameter_labels_with_superscripts_latex[i])

plt.show()

"""
This method shows the likelihood as a series of steps.
"""

fig, axes = plt.subplots(1, figsize=(10, 7))

for walker_index in range(search_internal.get_log_prob().shape[1]):
    axes.plot(search_internal.get_log_prob()[:, walker_index], alpha=0.3)

axes.set_ylabel("Log Likelihood")
axes.set_xlabel("step number")

plt.show()

"""
This method shows the parameter values of every walker at every step.
"""
fig, axes = plt.subplots(result.samples.model.prior_count, figsize=(10, 7), sharex=True)

for i in range(result.samples.model.prior_count):
    ax = axes[i]
    ax.plot(search_internal.get_chain()[:, :, i], alpha=0.3)
    ax.set_ylabel(result.model.parameter_labels_with_superscripts_latex[i])

axes[-1].set_xlabel("step number")

plt.show()

"""
Finish.
"""
