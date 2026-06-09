"""
Cookbook: Latent Variables
==========================

A latent variable is a quantity derived from the model parameters that is not itself sampled by the non-linear
search. Once the search has explored parameter space, every accepted sample can be transformed into a value of
the latent variable, building up a posterior distribution for it. This lets you report scientifically meaningful
quantities (with proper Bayesian uncertainties) without paying any cost during likelihood evaluation.

This cookbook is the foundational reference for latent variables across the **PyAuto** ecosystem. It explains
what latent variables ARE, why you would use them, how PyAutoFit computes them, what their errors really mean,
and how to load them downstream. The companion cookbooks `cookbooks/analysis.py` and `cookbooks/samples.py` show
the *writing* and *loading* APIs in code — this cookbook focuses on the concepts and cross-references them for
implementation details. The PyAutoGalaxy and PyAutoLens workspaces link to this script as their conceptual
starting point for latent variables.

__Contents__

 - Model Fit: Perform a model-fit using the example Gaussian + Analysis (which already ships a latent variable).
 - What is a Latent Variable?: The Bayesian framing — a deterministic function of the model parameters whose
   posterior is induced by the parameter posterior.
 - Why Latent Variables?: Three motivating cases — physical interpretability, derived quantities, aggregates.
 - How PyAutoFit Computes Latents: The `Latent.variables` hook and where in the search lifecycle it runs.
 - Two Output Modes: Every-sample versus N-draws-from-PDF, the `output.yaml` flags that toggle between them, and
   how to choose for your use case.
 - Errors on Latents: The 1σ / 3σ intervals are empirical quantiles of the *induced* latent posterior — not
   analytic Gaussian propagation. Why this matters for skewed or multi-modal models.
 - Posterior Draws Under the Hood: What `analysis.compute_latent_samples(result.samples)` actually does.
 - Loading Results Downstream: In-session access via the `Samples` API and out-of-session access via files /
   the aggregator.
 - When To Add A Latent vs A Sampled Parameter: A short rule-of-thumb.
"""

# from autoconf import setup_notebook; setup_notebook()

import autofit as af

from os import path

"""
__Model Fit__

We need a real fit to talk about latent samples concretely, so we run the standard noisy-1D-Gaussian example
shipped under `af.ex`. The dataset is auto-simulated if it doesn't already exist on disk.

`af.ex.Analysis` already defines a latent variable — the Gaussian's full-width half-maximum (FWHM) — via its
`Latent` class (`LatentExample`, which overrides `keys` and `variables`). The actual definitions live at
`PyAutoFit/autofit/example/analysis.py`. We re-use them here so this cookbook can focus entirely on
what latent variables ARE rather than how to declare them; the latter is the subject of `cookbooks/analysis.py`.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    name="cookbook_latent_variables",
    nwalkers=30,
    nsteps=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__What is a Latent Variable?__

In Bayesian language, a latent variable is a deterministic function `g(θ)` of the model parameters `θ`. The
non-linear search samples the joint posterior `p(θ | data)`. Because `g` is deterministic, every parameter sample
`θᵢ` carries a corresponding value `g(θᵢ)`, and the empirical distribution of those `g(θᵢ)` values is exactly
the induced posterior over the latent variable.

For the example Gaussian model, the FWHM is

  fwhm(σ) = 2 √(2 ln 2) · σ

`σ` is sampled directly by the search, but `fwhm` is not — it is derived after the fact. Yet because the posterior
over `σ` already exists, the posterior over `fwhm` is fully determined: just apply the formula to every accepted
sample.

The same trick generalises. Whenever a quantity you care about is a function of the parameters but not part of
the likelihood, it can be a latent variable.
"""

"""
__Why Latent Variables?__

Three common motivating cases:

 1. **Physical interpretability.** Scientists usually care about quantities the math doesn't sample directly. An
    astronomer probably wants the FWHM of a Gaussian rather than its standard deviation `σ`. A lens modeller
    wants the Einstein radius (a deflection-angle quantity) more than the SIE's `einstein_radius` parameter
    happens to mean. Adding a latent lets you report the science-friendly quantity with full Bayesian errors,
    without ever changing what the search actually fits.

 2. **Derived quantities.** Some quantities are too expensive or too messy to sample directly but can be
    computed from the fit afterwards. Magnification factors in strong lensing fall in this category — they
    require evaluating the lens model on a grid, which is cheap once the fit is done but cumbersome to fold
    into every likelihood call.

 3. **Aggregates.** When a model has many components (e.g. an MGE expansion of a galaxy's light into many
    Gaussians), per-component parameters are not by themselves scientifically meaningful. The total integrated
    flux is. Latents let you sum the contributions across components in a single named quantity.

Because latent variables are computed *after* the search completes, they cost zero per-likelihood evaluation —
adding one never slows the fit down.
"""

"""
__How PyAutoFit Computes Latents__

A model declares its latent variables with a `Latent` class (subclass of `af.Latent`), attached to the
`Analysis` via the `Latent` class attribute — just like `Visualizer` and `Result`:

 - `keys(analysis)` returns a list of dot-separated names. Each name becomes a column in `latent.csv` and a key
   in `latent_summary.json`. For the example Gaussian, `keys` returns `["gaussian.fwhm"]`.
 - `variables(analysis, parameters, model)` takes a parameter vector, builds the model instance from it, and
   returns a tuple of values — one per entry in `keys`, in the same order. It takes `parameters` (a raw vector)
   rather than `instance` so the function can be JAX-jit compiled and vmapped when the search runs under JAX.

Full API details (including how to write your own `Latent`) live in `cookbooks/analysis.py`. The library code
that drives the dispatch is at `PyAutoFit/autofit/non_linear/analysis/latent.py` (`latent_samples_from`).

The method is called *after* the search completes. It is never invoked inside `log_likelihood_function`, so
latent computations cannot influence which models the search prefers — they only attach extra information to
samples that have already been drawn.
"""

"""
__Two Output Modes__

PyAutoFit supports two strategies for computing latent values after a fit, controlled by `output.yaml`. Both
write a `latent_summary.json` containing the median and 1σ / 3σ intervals of every latent. They differ in
whether they also write the full per-sample table.

 - **N-draws-from-PDF mode** (`latent_draw_via_pdf: True`, the default):
   PyAutoFit draws `latent_draw_via_pdf_size` independent samples from the inferred posterior (default 100)
   and calls `Latent.variables` once per draw. Output is `latent_summary.json` only.
   *Fast*, because the cost scales with the draw count rather than the (typically much larger) number of
   accepted search samples. Suitable when you only need the latent's summary statistics.

 - **Every-sample mode** (`latent_draw_via_pdf: False`):
   PyAutoFit calls `Latent.variables` for every accepted sample of the non-linear search. Output is
   `latent_summary.json` *and* a full `latent/samples.csv` parallel to the parameter `samples.csv`.
   *Slow*, but gives you the complete latent posterior. Required if you want to plot a 2D corner of two
   latents jointly, or if you want to apply post-hoc filtering or weighting in the same way you would with
   the parameter samples.

Two further flags toggle whether latents are computed at all:

 - `latent_during_fit`: compute latents during the search (e.g. as part of incremental result updates). Off
   by default — slows down the search loop.
 - `latent_after_fit`: compute latents once the search finishes. On by default.

For most use cases the defaults are right: latents are computed once at the end, only summary statistics are
written, and you opt into every-sample mode when you specifically need it.
"""

"""
__Errors on Latents__

This is the most important concept in the cookbook, and the most commonly misunderstood. The 1σ and 3σ intervals
PyAutoFit writes for a latent variable are **empirical quantiles of the induced latent posterior**, not analytic
propagation of the parameter errors through `g`.

For a sampled parameter such as `σ`, the posterior is the marginal density `p(σ | data)`. The 1σ interval is
the central 68 percent of this marginal. PyAutoFit already does this for every parameter.

For a latent `g(θ)`, there is no marginal posterior to query directly. Instead, PyAutoFit transforms each
accepted sample `θᵢ` into `g(θᵢ)`, builds the empirical distribution of these values, and reads off the central
68 percent. The 1σ interval on the latent is the central 68 percent of that empirical distribution.

The crucial consequence: when the parameter posterior is non-Gaussian — skewed, multi-modal, or banana-shaped
— the latent posterior inherits that structure faithfully. There is no first-order Taylor expansion happening.
Asymmetric intervals on the latent are not bugs; they reflect real structure in the joint posterior.

This is one reason to prefer latents over "compute it analytically from the median parameter values" recipes.
The median of `g(θ)` is not in general `g(median(θ))`, and the symmetric `g(σ̄) ± dg/dσ · Δσ` interval can be
badly misleading. Latents always do it the honest way.

We can read the latent's 1σ interval back out via the same `Samples` API used for parameters. First, materialise
the latent samples from the fit result, then query them just like a regular `Samples` object:
"""
latent_samples = analysis.compute_latent_samples(result.samples)

median_instance = latent_samples.median_pdf()
print(f"Median PDF FWHM: {median_instance.gaussian.fwhm}")

max_likelihood_instance = latent_samples.max_log_likelihood()
print(f"Max log-likelihood FWHM: {max_likelihood_instance.gaussian.fwhm}")

"""
The intervals are asymmetric whenever the underlying parameter posterior is asymmetric — for instance, when the
prior on `σ` is bounded near zero. The 1σ interval is a pair `(lower, upper)` with no requirement that
`upper - median == median - lower`.
"""

"""
__Posterior Draws Under the Hood__

`analysis.compute_latent_samples(result.samples)` does mechanically what the explanation above describes:

 1. Choose the sample set — every accepted sample in every-sample mode, or `latent_draw_via_pdf_size` random
    draws from the posterior in N-draws mode.
 2. For each sample's parameter vector, call `Latent.variables(analysis, parameters, model)`.
 3. Pair each returned latent tuple with the original sample's `log_likelihood`, `log_prior`, and `weight`.
 4. Wrap the resulting collection in a `Samples` object whose API matches the original — `max_log_likelihood`,
    `median_pdf`, `values_at_sigma_1`, and so on all work as expected.

Each transformation is independent — the order doesn't matter, no parameter-to-parameter dependencies are
enforced. This is why JAX `jit` + `vmap` can accelerate the whole loop into a single batched compile when
the search runs under JAX.

If the latent computation itself is expensive (e.g. a multi-band lensing integral), the cost is what dominates,
not the dispatch loop. In that regime the N-draws-from-PDF mode is dramatically faster than every-sample mode
and almost always sufficient for reporting.
"""

"""
__Loading Results Downstream__

Once a fit has produced latent output, you can access it in two ways.

**In the same Python session**, materialise from the `Analysis` and `Result`:

```python
latent_samples = analysis.compute_latent_samples(result.samples)
print(latent_samples.median_pdf().gaussian.fwhm)
print(latent_samples.max_log_likelihood().gaussian.fwhm)
```

This is what the example above does. It recomputes the latents from the parameter samples on the fly — useful
if you want to add or change the latent definitions after the original fit ran, since you don't need to re-run
the non-linear search.

**After-the-fact via files**, the `latent/samples.csv` and `latent_summary.json` written to the output folder
are loaded by `paths.load_latent_samples()` and exposed via the same `Samples` API. The aggregator framework
also surfaces them for batched analysis across many fits. The end-to-end loading API is documented in
`cookbooks/samples.py` (look for the `__Derived Quantities__` section) and `cookbooks/result.py`.

The committed `latent.csv` file format is intentionally human-readable — one column per latent key,
one row per sample (or per posterior draw, in N-draws mode), plus the standard `log_likelihood`, `log_prior`,
and `weight` columns. You can open it in any spreadsheet tool for a quick visual check.
"""

"""
__When To Add A Latent vs A Sampled Parameter__

A short rule of thumb:

 - If the quantity appears in your `log_likelihood_function` — make it a model parameter. The non-linear
   search needs to explore it to compute the likelihood.

 - If the quantity is purely a derived interpretation of the parameters (a function of them, with no impact
   on the likelihood) — make it a latent. Latents are free at search time, expressive at report time, and
   their Bayesian errors are exactly the right thing.

When in doubt, prefer latents. They are inexpensive to add and never bias the search; the cost of getting it
wrong is essentially zero. The cost of mistakenly inflating the parameter vector with redundant variables is
much higher — the search has to explore the extra dimensions, and degeneracies between the redundant variable
and the original ones can make the search markedly slower.

Latent variables shine when downstream consumers (your collaborators, your readers, your aggregator pipelines)
need to compare across many fits. A single latent column in `latent.csv` makes that comparison trivial; deriving
the same quantity by hand from `samples.csv` is error-prone and verbose.
"""
