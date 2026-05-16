The `searches` folder contains example scripts which fit a model using the different non-linear searches supported by **PyAutoFit**:

# Files

- `nest.py`: Fit a model using nested sampling algorithms (Dynesty static, Dynesty dynamic, Nautilus, NSS).
- `mcmc.py`: Fit a model using MCMC algorithms (Emcee, Zeus).
- `mle.py`: Fit a model using maximum likelihood algorithms (Drawer, LBFGS).
- `start_point.py`: Set the start-point of certain parameters in a model-fit.

# When to use NSS

`af.NSS` (Nested Slice Sampling, JAX-native) joins the nested-sampling lineup in `nest.py`. The boundary-based samplers (Dynesty, Nautilus) work with any Python log-likelihood. NSS goes further and runs its inner sampling loop entirely inside `jax.jit` — when your log-likelihood is itself JAX-traceable, the per-evaluation cost drops by roughly an order of magnitude versus the boundary samplers.

Use NSS if:

- Your `Analysis.log_likelihood_function` is JAX-traceable (autolens / autogalaxy MGE + pixelization pipelines and most autofit examples qualify when the analysis is constructed with `use_jax=True`).
- You want SLURM-friendly automatic resume — NSS writes a checkpoint every `checkpoint_interval` outer iterations and recovers from it on the next run with the same Paths.
- You want on-the-fly visualization during long fits — set `iterations_per_quick_update` and `analysis.visualize` will be called with the current best live point between outer iterations.

Stick with Nautilus / Dynesty if your log-likelihood is pure-NumPy or includes scipy fallbacks that don't JIT cleanly. All three samplers expose the same `result.samples` interface, so swapping between them is one line.

Install: `pip install autofit[nss]` (single command — see the `Install Precondition for NSS` section in `nest.py`).
