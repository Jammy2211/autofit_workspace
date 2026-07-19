#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

if [ "$PYTHON_VERSION" = "3.12" ]; then
  pip install ./PyAutoNerves "./PyAutoFit[optional]"
else
  pip install ./PyAutoNerves ./PyAutoFit
fi
pip install nautilus-sampler
# NSS sampler — searches/nest.py exercises `af.NSS`, which needs the
# `[nss]` extra plus two manually-installed git pins. The SHAs match
# PyAutoFit/pyproject.toml `[project.optional-dependencies] nss`.
pip install "./PyAutoFit[nss]"
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@ef45acd2f2fa0cca15adbdcd3ff7cb3a98987cb5"
pip install "nss @ git+https://github.com/yallup/nss.git@69159b0f4a3a53123b9eec7df91e4ed3885e4dc4"
pip install jupyter nbconvert ipynb-py-convert
