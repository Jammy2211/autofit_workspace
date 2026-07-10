"""
Feature: Expectation Propagation (Step-by-Step)
===============================================

The `graphical_models.py` feature example composed a graphical model from many datasets and fitted it with a single
non-linear search over the joint parameter space. That approach hits a ceiling: with tens of datasets the joint
parameter space becomes too high dimensional to sample.

Expectation propagation (EP) is **PyAutoFit**'s answer. Instead of one high-dimensional fit, EP fits the factor graph
one node at a time — each dataset is fitted on its own, at low dimensionality, and the fits exchange "messages" that
carry what each fit learned about shared parameters. Cycling through the datasets a few times converges to an
approximation of the full joint posterior.

Most users run EP through the high-level API (`factor_graph.optimise(...)`, as shown at the end of this script and in
the `HowToFit` chapter 3 tutorials). This example instead **dissects one EP update at the low-level API**, so you can
run the machinery step by step and see exactly what the statistics are doing at every stage: the mean-field
approximation, the cavity distribution, the tilted-distribution fit, moment matching and the damped message update.

The mathematics implemented by each step is stated formally (with the same equation numbering) in the **PyAutoFit**
source: `autofit/graphical/README.md`. This script is the runnable companion to that document.

__Contents__

This script is split into the following sections:

- **Example Source Code (`af.ex`)**: The example objects used in this script.
- **Dataset**: Load 3 noisy 1D Gaussian datasets which share a common centre.
- **Model**: Compose the graphical model with a shared `centre` prior.
- **Factor Graph**: Pair each model with its dataset's Analysis and build the factor graph.
- **Mean Field**: The EP approximation — one exponential-family message per (factor, variable).
- **Cavity**: Pull one factor out of the approximation and inspect its cavity distribution.
- **Tilted Fit**: Fit that factor against its cavity (the tilted distribution) with a Laplace optimiser.
- **Damped Update**: Divide out the cavity and update the factor's messages, with damping.
- **Convergence**: Measure the change with a KL divergence — the quantity EP's stopping criterion watches.
- **Full EP Fit**: Run the same loop end-to-end via the high-level API and inspect the results.
- **Wrap Up**: Where to go next.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this script:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects used elsewhere in the workspace.
"""

# from autoconf import setup_notebook; setup_notebook()

from os import path

import autofit as af

"""
__Dataset__

We load 3 noisy 1D Gaussian datasets (`gaussian_x1_0`, `gaussian_x1_1`, `gaussian_x1_2`). All three were simulated
with the same `centre=50.0` but different widths — exactly the situation graphical models are for: a parameter that
is global across datasets, which every dataset constrains and the graph should combine.

If the datasets do not exist on your system they are created by running the corresponding simulator script.
"""
total_datasets = 3

if not path.exists(path.join("dataset", "example_1d", "gaussian_x1_0")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1_{dataset_index}"
    )

    data_list.append(
        af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    )
    noise_map_list.append(
        af.util.numpy_array_from_json(
            file_path=path.join(dataset_path, "noise_map.json")
        )
    )

"""
__Model__

Every dataset gets its own `Gaussian` model, but all of them share a single prior object for `centre`. Sharing the
prior *object* (not just its settings) is what tells **PyAutoFit** the parameter is global: on the factor graph, the
shared prior becomes one variable that three factors connect to.

The priors are deliberately centred *away* from the true values (the datasets were simulated with `centre=50.0`), so
you can watch EP pull the approximation from the prior towards the data at every step below.
"""
centre_shared_prior = af.GaussianPrior(mean=35.0, sigma=30.0)

model_list = []

for model_index in range(total_datasets):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # one object, shared by all datasets
    gaussian.normalization = af.GaussianPrior(mean=15.0, sigma=10.0)
    gaussian.sigma = af.GaussianPrior(mean=5.0, sigma=10.0)

    model_list.append(gaussian)

"""
__Factor Graph__

The joint model EP will approximate is a product of factors [README Eq. (1)]:

    p(x) ∝ ∏ₐ fₐ(xₐ)

Each dataset contributes one likelihood factor — an `AnalysisFactor` pairing its model with its `Analysis` — and, as
we are about to see, every prior contributes a factor of its own too.

Because EP fits the factor graph one node at a time, **each `AnalysisFactor` carries its own non-linear search**,
which will be used to fit that node's tilted distribution (defined below). We use a `DynestyStatic` with a small
budget — each fit is only 3-dimensional, one of EP's key selling points.
"""
search = af.DynestyStatic(
    nlive=150,
    sample="rwalk",
    walks=10,
)

analysis_factor_list = []

for dataset_index, (model, data, noise_map) in enumerate(
    zip(model_list, data_list, noise_map_list)
):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_factor_list.append(
        af.AnalysisFactor(
            prior_model=model,
            analysis=analysis,
            optimiser=search,
            name=f"dataset_{dataset_index}",
        )
    )

factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Mean Field__

EP approximates the posterior with a fully factorised ("mean field") distribution [README Eq. (2)]:

    q(x) = ∏ₐ qₐ(x),    qₐ(x) = ∏ᵢ qₐᵢ(xᵢ)

Every factor `a` owns one **message** `qₐᵢ` per variable `i` it touches — an exponential-family distribution
(`autofit/messages/`). The `EPMeanField` object holds the whole approximation: a dictionary mapping every factor to
its `MeanField` (its dictionary of messages).

At initialisation each message is just the variable's prior.
"""
model_approx = factor_graph.mean_field_approximation()

print("The factors on the graph:")
for factor in model_approx.factor_graph.factors:
    print(f"  {factor.name}")

"""
Note the factor count: 3 `AnalysisFactor`'s (the datasets) **plus 7 `PriorFactor`'s** — one per free parameter
(1 shared centre + 3x2 per-dataset parameters). On a factor graph, priors are not special: each prior is simply one
more factor multiplying the joint distribution, and it participates in message passing like any other.

The global approximation for any single variable is the product of every factor's message on it. Because messages are
exponential-family distributions, products just add natural parameters [README Eq. (4)] — this is why EP is cheap.

Let's look at the shared centre's current global approximation and its natural parameters (η₁ = μ/σ², η₂ = −1/(2σ²)
for a Gaussian message):
"""
print("Shared centre, global approximation at initialisation:")
print(f"  {model_approx.mean_field[centre_shared_prior]}")
print(f"  natural parameters: {model_approx.mean_field[centre_shared_prior].natural_parameters()}")

"""
__Cavity__

One EP step updates one factor. We pull the first dataset's factor out of the approximation
(`factor_approximation`), which splits the mean field into [README Eq. (5)]:

 - `factor_dist` — the messages *this factor* currently owns, `qₐ`.
 - `cavity_dist` — the product of *every other factor's* messages on this factor's variables, `q⁻ᵃ`. On natural
   parameters: η_cav = Σ_{b≠a} η_b.
 - `model_dist` — their product, the full current approximation restricted to this factor's variables.

Statistically, the cavity is "everything the rest of the graph believes about this factor's parameters" — it will act
as the *prior* for this factor's fit. This is the message-passing step: the other datasets' inferences about the
shared centre arrive here, encoded in the cavity.
"""
factor = model_approx.factor_graph.factors[0]
factor_approx = model_approx.factor_approximation(factor)

print(f"Cavity distribution of {factor.name} for the shared centre:")
print(f"  {factor_approx.cavity_dist[centre_shared_prior]}")
print(f"Factor's own message for the shared centre:")
print(f"  {factor_approx.factor_dist[centre_shared_prior]}")

"""
At initialisation the factor's own message on the centre is (close to) flat and the cavity is dominated by the shared
prior's `PriorFactor` — after a full EP cycle, the cavity will instead carry the other two datasets' posteriors.

__Tilted Fit__

The **tilted distribution** is this factor's exact likelihood times its cavity [README Eq. (6)]:

    p̂ₐ(x) = fₐ(x) q⁻ᵃ(x) / Ẑₐ

This is a proper low-dimensional posterior: likelihood = this dataset, prior = the cavity. Fitting it is an ordinary
**PyAutoFit** fit — which is why each `AnalysisFactor` carries its own search. Behind the scenes, the cavity messages
are installed as the model's priors, the search samples the tilted posterior, and the weighted samples are
**moment matched**: the exponential-family member whose expected sufficient statistics equal the samples' is the
KL-optimal projection [README Eqs. (7)-(9)]. (A `LaplaceOptimiser` can be used instead of a sampler, replacing
sampling + moment matching with a Gaussian at the mode with covariance from the curvature.)

The result is a new `MeanField` over this factor's variables — the projected tilted posterior q*. Watch the shared
centre move from the prior (35) towards the truth (50):
"""
new_model_dist, status = search.optimise(factor_approx)

print(f"Projected tilted posterior (q*) for the shared centre after fitting {factor.name}:")
print(f"  {new_model_dist[centre_shared_prior]}")
print(f"Fit status: success={status.success}")

"""
The centre's sigma has shrunk relative to the cavity: this dataset has added its information.

__Damped Update__

q* is the new belief about *all* factors combined (it was fitted against the cavity). To extract what *this factor
alone* learned, EP divides the cavity back out, with damping δ ∈ (0, 1] [README Eqs. (10)-(11)]:

    qₐ_new = (q*)^δ (qₐ_old)^(1−δ) / (q⁻ᵃ)^δ

On natural parameters this is an exponential moving average — for δ=1 it is plain division, smaller δ takes a
partial step (damping stabilises EP on graphs where factors disagree):

    ηₐ ← (1 − δ) ηₐ + δ (η_q* − η_cav)

`project_mean_field` performs this update and returns a *new* `EPMeanField` (the old one is unchanged):
"""
model_approx_new, status = model_approx.project_mean_field(
    new_model_dist, factor_approx, delta=0.7
)

print("Shared centre, global approximation:")
print(f"  before: {model_approx.mean_field[centre_shared_prior]}")
print(f"  after:  {model_approx_new.mean_field[centre_shared_prior]}")

"""
Note the division is performed on natural parameters, which is not guaranteed to give a valid distribution (e.g. it
can produce a negative variance when the new fit is *less* certain than the cavity). When that happens **PyAutoFit**
falls back to the factor's previous message for the offending parameters and records a `BAD_PROJECTION` status —
damping makes this rarer.

__Convergence__

How much did that one update change our approximation? EP measures this with the KL divergence between the new and
old mean fields, summed over variables [README Eq. (12)] — exactly the quantity `af.EPHistory` watches to decide
termination (stop when it drops below `kl_tol`):
"""
kl = model_approx_new.mean_field.kl(model_approx.mean_field)

print(f"KL(new || old) after one factor update: {kl}")

"""
A full EP iteration repeats the cavity → tilted fit → damped update cycle for every factor on the graph (including
the `PriorFactor`'s, whose updates are analytic — a prior that is already exponential-family multiplies in exactly,
no search needed). Iterations repeat until the KL change converges.

__Full EP Fit__

That is the entire algorithm. The high-level API runs this loop for us: `factor_graph.optimise(...)` cycles the
factors, manages damping, tracks the history and writes results to the output folder.

Each `AnalysisFactor` is fitted by its own `DynestyStatic`; the `LaplaceOptimiser` passed here is the *default* for
the remaining factors — the `PriorFactor`'s — whose updates are cheap.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join("features", "expectation_propagation")
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace,
    paths=paths,
    ep_history=af.EPHistory(kl_tol=0.05),
    max_steps=5,
)

"""
The result contains the converged mean field — the EP approximation to the joint posterior. Its `mean`, `variance`
and `scale` dictionaries give the marginal posterior of every parameter; the shared centre's entry is the global
inference all three datasets contributed to.
"""
mean_field = factor_graph_result.updated_ep_mean_field.mean_field

print("Converged posterior approximation:")
print(f"  means:  {mean_field.mean}")
print(f"  scales: {mean_field.scale}")

"""
__Output Folder__

The fit above wrote to `output/features/expectation_propagation`. Notable contents:

 - `graph.info`: the composed graphical model — every parameter, how parameters are shared across factors, and the
   priors on each.

 - One folder per `AnalysisFactor` containing `optimization_#` sub-folders — the repeated per-dataset fits across EP
   cycles. The `model.info` files show the priors (i.e. cavities) updating cycle to cycle.

 - `graph.results`: the current mean-field summary.

__Wrap Up__

You have now run every statistical step of EP by hand: the mean-field approximation, the cavity, the tilted fit,
moment matching, the damped natural-parameter update and the KL convergence check.

Where to go next:

 - `autofit/graphical/README.md` (in the **PyAutoFit** source) — the formal equations behind every step above, with
   pointers into the implementation.

 - `features/graphical_models.py` — composing richer graphical models (this script's prerequisite reading).

 - **HowToFit** chapter 3 (`scripts/chapter_3_graphical_models/`) — the full tutorial series, including hierarchical
   models where shared parameters are drawn from a parent distribution, and EP fits of those hierarchies.
"""
