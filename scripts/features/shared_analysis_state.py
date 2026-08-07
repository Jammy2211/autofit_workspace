"""
Feature: Shared Analysis State
==============================

The `graphical_models.py` feature example showed how to fit many datasets simultaneously with a `FactorGraphModel`,
where each dataset has its own `Analysis` (and therefore its own `log_likelihood_function`) and the factors are linked
by shared priors. When a `FactorGraphModel` evaluates its likelihood it loops over every factor, calls each factor's
`log_likelihood_function`, and sums the results.

For some problems a large fraction of the work done inside each factor's `log_likelihood_function` is *identical* across
every factor, because the factors share model parameters. Recomputing it once per factor is then pure waste. This
example shows how to compute that shared work **once per likelihood evaluation** and reuse it across every factor, using
the `Analysis.shared_state_from` hook.

__The motivating problem__

Consider fitting `N` datasets that share the *entire* model — for example the same astronomical source observed by `N`
instruments, or the `N` spectral channels of a datacube that share a single physical model. Every factor has the same
model, so the **model data is identical for every factor**; only the data each factor is compared against differs.

Without sharing, each of the `N` factors rebuilds the same model data from scratch every likelihood evaluation. If the
model data is expensive to compute, the per-evaluation cost scales as `N x (expensive model-data build)` when it could
be `1 x (expensive model-data build) + N x (cheap comparison)`.

This is exactly the structure of the strong-lensing datacube likelihood that motivated this feature: `N` spectral
channels share one lens model, so the expensive shared work (ray-tracing the lens model, building the source-plane
pixelization and its mapping and curvature matrices) is identical for every channel and dominates the per-channel cost.
The 1D Gaussian below stands in for that expensive shared computation.

__When sharing is valid__

Sharing model data is only correct when the model data really is identical for every factor — i.e. when the *whole*
model is shared. Contrast this with `graphical_models.py`, where only the `centre` was shared and each `Gaussian` had its
own `normalization` and `sigma`: there the model data differs between factors, so sharing it would silently produce the
wrong likelihood. The mechanism is therefore strictly opt-in (`share_model_data=True`), and you opt in only when you know
the shared work is genuinely invariant across factors.

__Contents__

- **Example Source Code (`af.ex`)**: The example objects used in this script.
- **Dataset**: Load 3 noisy 1D Gaussian datasets, here treated as repeat observations of one shared model.
- **Analysis**: Create `Analysis` objects that opt into the shared-state mechanism.
- **Shared State**: How `Analysis.shared_state_from` and the `shared` argument of `log_likelihood_function` fit together.
- **Model**: Compose a model whose parameters are *fully* shared across every factor.
- **Analysis Factors / Factor Graph**: Build the factor graph, exactly as in `graphical_models.py`.
- **Search**: Fit the factor graph; the shared model data is now computed once per evaluation, not once per factor.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and `visualize`
 functions. In this example we use its `share_model_data` option and its `shared_state_from` method, which demonstrate
 the shared-state mechanism on the 1D Gaussian toy.

 - `Gaussian`: a model component representing a 1D Gaussian profile.
"""

# from autofit import setup_notebook; setup_notebook()

from os import path
import matplotlib.pyplot as plt

import autofit as af

"""
__Dataset__

We load 3 noisy 1D Gaussian datasets. In `graphical_models.py` these were three different Gaussians that happened to
share a `centre`. Here we instead treat them as three observations that share the *entire* model — the regime in which
sharing the model data is valid.
"""
total_datasets = 3

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding simulator script.
"""
# Intentional raw guard: this guards a single file, and this workspace has no
# should_simulate / PYAUTO_SMALL_DATASETS contract (unlike autolens/autogalaxy).
if not path.exists(
    path.join("dataset", "example_1d", "gaussian_x1__low_snr", "dataset_0", "data.json")
):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators_sample.py"],
        check=True,
    )

data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", f"dataset_{dataset_index}"
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)

for dataset_index in range(total_datasets):
    xvalues = range(data_list[dataset_index].shape[0])

    plt.errorbar(
        x=xvalues,
        y=data_list[dataset_index],
        yerr=noise_map_list[dataset_index],
        linestyle="",
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Data.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

"""
__Analysis__

We create an `Analysis` for each dataset, exactly as in `graphical_models.py`, but we pass `share_model_data=True`.

This opts the `Analysis` into the shared-state mechanism: it tells **PyAutoFit** that the model is fully shared across
every factor, so the model data is identical for all of them and need only be computed once. With the default
(`share_model_data=False`) the `Analysis` behaves exactly as in every other example.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, share_model_data=True)

    analysis_list.append(analysis)

"""
__Shared State__

Two pieces of the `af.ex.Analysis` API make the sharing work, and they are the general hooks any `Analysis` can use:

 - `shared_state_from(instance)`: computes the object that is shared across factors. For the 1D Gaussian this is simply
 the model data (the Gaussian evaluated on the grid). It returns `None` unless `share_model_data=True`, so by default no
 state is shared. This is the per-evaluation, cross-factor sibling of `modify_before_fit`: where `modify_before_fit`
 precomputes state *once before the fit* that does not depend on the model, `shared_state_from` precomputes state *once
 per likelihood evaluation* that does depend on the model (the parameters change every sample).

 - `log_likelihood_function(instance, shared=None)`: gains an optional `shared` argument. When the factor graph passes a
 `shared` object the analysis uses it directly instead of recomputing the model data; when `shared` is `None` (e.g. a
 standalone single-dataset fit) it computes the model data itself, so the standalone behaviour is unchanged.

When the `FactorGraphModel` evaluates its likelihood it calls `shared_state_from` once, on the lead factor, and forwards
the result as `shared=` to every factor's `log_likelihood_function`. Because the model is fully shared, the lead
factor's model data is identical to what every other factor would have computed, so reusing it is exact. If no factor
opts in, the graph calls each `log_likelihood_function` exactly as before and nothing changes.

__Model__

We now compose the model. The key difference from `graphical_models.py` is that *every* parameter is shared: we build a
single set of priors and pass them to a `Gaussian` for each dataset, so all factors use identical `centre`,
`normalization` and `sigma`. This is what makes the model data identical across factors, and therefore what makes
sharing it valid.
"""
centre = af.GaussianPrior(mean=50.0, sigma=30.0)
normalization = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

model_list = []

for model_index in range(total_datasets):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre  # All three priors are shared across every Gaussian,
    gaussian.normalization = normalization  # so the model — and therefore the model
    gaussian.sigma = sigma  # data — is identical for every factor.

    model_list.append(gaussian)

"""
__Analysis Factors__

We pair each model with its `Analysis`, exactly as in `graphical_models.py`.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We combine the `AnalysisFactor`'s into a factor graph. Nothing about building the graph changes when using shared state
— the sharing is entirely a property of the `Analysis` objects.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Search__

We fit the factor graph with a non-linear search, using its `global_prior_model` property.

Every time the search evaluates the likelihood, the shared model data is now computed **once** and reused by all three
factors, instead of being rebuilt three times. For the 1D Gaussian the saving is negligible, but for an expensive model
data — such as the lensing datacube that motivated this feature — collapsing `N` rebuilds into one is the difference
between an intractable fit and a fast one.
"""
search = af.DynestyStatic(
    path_prefix="features", name="shared_analysis_state", sample="rwalk"
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Wrap Up__

This example introduced the `Analysis.shared_state_from` hook and the `shared` argument of `log_likelihood_function`,
the general mechanism by which the factors of a `FactorGraphModel` share per-evaluation, model-dependent state.

The mechanism is domain-agnostic: **PyAutoFit** knows only "a factor may want to compute an object once and have every
factor see it". What that object is, and the contract that it really is shared, are entirely the responsibility of the
`Analysis` you write — here, the 1D Gaussian model data, but in the motivating science case the lens model's ray-traced
grids, source-plane pixelization and curvature matrix.

Remember the correctness contract: only opt in (`share_model_data=True` here, or a non-`None` `shared_state_from` in
general) when the shared work is genuinely identical across every factor. When only some parameters are shared, as in
`graphical_models.py`, leave it off and let each factor compute its own state.
"""
