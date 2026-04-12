# Copilot Coding Agent Instructions

You are working on the **autofit_workspace**, a tutorial/example repository for the PyAutoFit library.

## Key Rules

- Only edit files in `scripts/`. Never edit files in `notebooks/` (those are auto-generated).
- After making changes, always test with `bash run_scripts.sh`.
- If tests fail, read the log in `failed/<path>.log`, fix the script, and re-run.
- Do not add new scripts. Only update existing ones to match API changes.
- Preserve all docstrings, comments, and tutorial explanations. Only change code that uses the old API.

## Testing

`run_scripts.sh` sets `PYAUTO_TEST_MODE=1` automatically. Every script should pass in this mode. A script that fails in test mode indicates a real problem (broken import, wrong function name, etc.).

## Notebook Generation

After all scripts pass testing, regenerate the notebooks:

```bash
pip install ipynb-py-convert
git clone https://github.com/Jammy2211/PyAutoBuild.git ../PyAutoBuild
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autofit
```

Run this from the workspace root. Commit the regenerated notebooks alongside the script changes.

## PR Description

When opening your PR, include:
- A summary of which APIs changed and how
- A list of all scripts you updated
- Confirmation that notebooks were regenerated
- A "Could not update" section for any scripts that still fail, with the error and your assessment of why
