# PyAutoFit Workspace — Agent Instructions

This is the tutorial and example workspace for **PyAutoFit**. These are the canonical,
agent-agnostic instructions for this repo. PyAutoFit is a general-purpose Python framework for
Bayesian model fitting and inference — model composition, priors, non-linear searches, samples,
the aggregator, and graphical models. It is **not** tied to any science domain; the examples here
are deliberately domain-agnostic toy problems (fitting 1D profiles such as Gaussians).

## Repository Structure

- `scripts/` — Runnable Python scripts, organised by PyAutoFit feature:
  - `overview/` — Concise tour of the core features (`overview_1_the_basics.py` first).
  - `searches/` — Every non-linear search: `nest.py` (dynesty, nautilus), `mcmc.py` (emcee,
    zeus), `mle.py` (drawer, LBFGS), `start_point.py`.
  - `model/` — Model composition (priors, collections, custom model objects).
  - `features/` — Advanced features: graphical models, model comparison, search chaining,
    search grid search, sensitivity mapping, interpolation, shared analysis state.
  - `cookbooks/` — Focused API cookbooks (model, result, samples, latent variables, …).
  - `plot/` — Plotting search results (dynesty, emcee, nautilus, zeus, GetDist).
  - `simulators/` — Scripts that simulate the example datasets fitted throughout.
- `notebooks/` — Jupyter notebook versions, generated from `scripts/` (do not edit directly).
- `config/` — PyAutoFit configuration YAML files (`config/general.yaml` controls search update
  intervals, output, logging, profiling, parallel warnings).
- `dataset/` — Example 1D datasets.
- `output/` — Non-linear search results (generated at runtime, not committed).

There is no `projects/` directory.

## Running Scripts

Scripts are run **from the repository root** so relative paths to `dataset/`, `config/`, and
`output/` resolve correctly:

```bash
python scripts/overview/overview_1_the_basics.py
```

There is currently no `start_here.py` in this workspace — `scripts/overview/overview_1_the_basics.py`
is the canonical entry point and the always-current end-to-end reference. (The runner still honours
the convention: *if* a folder contains a `start_here.py`, it executes that first, before other
scripts and subfolders, since some scripts depend on results it produces.)

### Standard imports

```python
import autofit as af
import autofit.plot as aplt
```

For the canonical workflow (compose a model → configure a search → define an analysis → fit →
inspect results), read `scripts/overview/overview_1_the_basics.py` — it is kept current with the API.

## Testing

On CI, every PR is gated by three workflows on Python **3.12 and 3.13**: `smoke_tests.yml` (the
smoke runner below — the definition of green), `navigator_check.yml` (PyAutoBuild's reusable
navigator-catalogue check; see *Notebooks vs Scripts*), and `url_check.yml` (link checking). The
smoke and navigator jobs check out **PyAutoBuild** as a sibling and run the PyAuto* libraries from
the **same-named branch** of each source repo, so a workspace PR is validated against matching
library branches.

Two runners exist; both are run from the repo root.

**Full runner** — executes every script in `scripts/`:

```bash
bash run_scripts.sh
```

It exports `PYAUTO_TEST_MODE=1` (plus skip/fast flags), runs each script, writes a log for every
failure to `failed/<relative_path>.log`, continues past failures, and prints a final
`Results: P passed, F failed, S skipped` line. The skip list comes from
`../PyAutoBuild/autobuild/config/no_run.yaml` under the `autofit:` key; entries whose comment
mentions a bug/issue/fix are flagged `[TODO - should run after a future PR]`.

**Smoke runner** — executes only the curated subset:

```bash
python .github/scripts/run_smoke.py
```

It is driven by `smoke_tests.txt` (scripts) and `smoke_notebooks.txt` (notebooks) in the workspace
root, with per-entry environment from `config/build/env_vars.yaml`. It prints a `[PASS]` /
`[FAIL (exit N)]` line per entry, ends with a `=== Smoke test summary: P/T passed ===` line, and
exits non-zero if any entry failed.

`PYAUTO_TEST_MODE` is PyAutoFit's own environment variable: it tells the non-linear search to skip
its actual sampling, turning a run into a fast structural / end-to-end check that the model composes
and the script runs to completion — without paying for inference. A script that fails under it
signals a real problem (broken import, renamed API, etc.).

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to their default cache locations, point them at writable
directories:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/overview/overview_1_the_basics.py
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are **generated** from the `.py` files in `scripts/`. **Always edit the
`.py` scripts, never the `.ipynb` notebooks directly.** The `# %%` markers alternate between code
and markdown cells.

### Generating notebooks

After updating scripts, regenerate the notebooks using the PyAutoBuild tool (run from the workspace
root):

```bash
pip install ipynb-py-convert
git clone https://github.com/PyAutoLabs/PyAutoBuild.git ../PyAutoBuild
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autofit
```

Commit the regenerated notebooks alongside the script changes. The `/generate_and_merge` skill
automates build + commit + PR + merge.

## Bulk-edit safety

When editing the same region across many scripts in one pass (adding a section, renaming a symbol,
updating an import block), only rewrite the targeted region. **Never produce a whole-file write
unless you have read the entire current contents of that file** — a whole-file write based on a
header skim silently deletes every section below the header. Prefer targeted edits.

## Methodological context

PyAutoFit is domain-agnostic, so there is no single science field to reference. Any "context" for
its examples is statistical / methodological — Bayesian inference, priors, non-linear search,
model comparison — not a scientific subject. Do not graft lensing, galaxy, or other domain concepts
onto this workspace.

## Related Repos

PyAutoFit sits near the base of the PyAuto stack (all on the `PyAutoLabs` GitHub org):

- https://github.com/PyAutoLabs/PyAutoConf — configuration handling (the `autoconf` dependency).
- https://github.com/PyAutoLabs/PyAutoFit — this library: model composition + non-linear search.
- https://github.com/PyAutoLabs/PyAutoBuild — notebook generation + CI.
- https://github.com/PyAutoLabs/PyAutoGalaxy — downstream science library built **on** PyAutoFit.
- https://github.com/PyAutoLabs/PyAutoLens — downstream science library built **on** PyAutoFit.
- https://github.com/PyAutoLabs/HowToFit — tutorial lecture series that teaches statistical
  model-fitting and Bayesian inference with PyAutoFit from first principles; the starting point for
  beginners new to the framework.

For local development these are typically cloned as siblings of this repo (`../PyAutoFit`,
`../PyAutoConf`, `../PyAutoBuild`, …).

## Task Workflows

### API Update tasks

When assigned an issue titled `[API Update]`:

1. Read the PR diff in the issue body. Identify every renamed, moved, removed, or changed public
   API (functions, classes, method signatures, parameter names, import paths).
2. Search **all** `.py` files in `scripts/` for usages of the old API.
3. Update each file to the new API, preserving existing behaviour, docstrings, and comments.
4. Run `bash run_scripts.sh` from the repo root to test (use `python .github/scripts/run_smoke.py`
   for a fast subset).
5. Read any `failed/<path>.log` files and fix the affected scripts.
6. Repeat 4–5 until the suite is clean.
7. If a script cannot be fixed (ambiguous change, missing dependency), leave it unchanged and list
   it in the PR description under **"Could not update"** with the reason.
8. After all scripts pass, regenerate the notebooks (see "Generating notebooks").

### General Issue tasks

When assigned a general (non-API) issue:

1. Read the issue description and any linked plan or AI prompt.
2. Identify which scripts need to be created or modified.
3. Only edit files in `scripts/`. Never edit `notebooks/` directly.
4. Preserve all docstrings, comments, and tutorial explanations.
5. Test with `bash run_scripts.sh` after changes.
6. Regenerate notebooks after all scripts pass.

### PR description

When opening your PR, include:

- A summary of what changed and why.
- A list of all scripts you updated or created.
- Confirmation that notebooks were regenerated.
- A "Could not update" section for any scripts that still fail, with the error and your assessment.

<!-- repos_sync:history:begin -->
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
<!-- repos_sync:history:end -->
