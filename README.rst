PyAutoFit Workspace
====================

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.4.13.6/start_here.ipynb

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02550/status.svg
   :target: https://doi.org/10.21105/joss.02550

|colab| |JOSS|

`Installation Guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautofit.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Colab <https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.4.13.6/notebooks/overview/overview_1_the_basics.ipynb>`_ |
`HowToFit <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_

Welcome to the **PyAutoFit** Workspace! 

Getting Started
---------------

You can get set up on your personal computer by following the installation guide on
our `readthedocs <https://pyautofit.readthedocs.io/>`_.

Alternatively, you can try **PyAutoFit** out in a web browser by going to the `autofit workspace
Colab <https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.4.13.6/notebooks/overview/overview_1_the_basics.ipynb>`_.

Where To Go?
------------

We recommend that you start with the ``autofit_workspace/notebooks/overview/overview_1_the_basics.ipynb``
notebook, which will give you a concise overview of **PyAutoFit**'s core features and API.

Next, read through the overview example notebooks of features you are interested in, in the folder: ``autofit_workspace/notebooks/overview``.

Then, you may wish to implement your own model in **PyAutoFit**, using the ``cookbooks`` for help with the API. Alternative,
you may want to checkout the ``features`` package for a list of advanced statistical modeling features.

HowToFit
--------

For users less familiar with Bayesian inference and scientific analysis you may wish to read through
the **HowToFits** lectures. These teach you the basic principles of Bayesian inference, with the
content pitched at undergraduate level and above.

A complete overview of the lectures `is provided on the HowToFit readthedocs page <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.htmll>`_

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoFit** examples written as Jupyter notebooks.
- ``scipts``: **PyAutoFit** examples written as Python scripts.
- ``projects``: Example projects which use **PyAutoFit**, which serve as a illustration of model-fitting problems and the **PyAutoFit** API.
- ``config``: Configuration files which customize **PyAutoFit**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed with **PyAutoFit**.
- ``output``: Where the **PyAutoFit** analysis and visualization are output.

The **examples** in the notebooks and scripts folders are structured as follows:

- ``overview``: Examples using **PyAutoFit** to compose and fit a model to data via a non-linear search.
- ``cookbooks``: Concise API reference guides for **PyAutoFit**'s core features.
- ``features``: Examples of **PyAutoFit**'s advanced modeling features.
- ``howtofit``: Detailed step-by-step tutorials.
- ``searches``: Example scripts of every non-linear search supported by **PyAutoFit**.
- ``plot``: An API reference guide for **PyAutoFits**'s plotting tools.

The following **projects** are available in the project folder:

- ``astro``: An Astronomy project which fits images of gravitationally lensed galaxies.

Workspace Version
-----------------

This version of the workspace are built and tested for using **PyAutoFit v2026.4.5.3**.

Support
-------

Support for installation issues and integrating your modeling software with **PyAutoFit** is available by
`raising an issue on the autofit_workspace GitHub page <https://github.com/Jammy2211/autofit_workspace/issues>`_. or
joining the **PyAutoFit** `Slack channel <https://pyautofit.slack.com/>`_, where we also provide the latest updates on
**PyAutoFit**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.

Build Configuration
-------------------

The ``config/`` directory contains two files used by the automated build and test system
(CI, smoke tests, and pre-release checks). These are not relevant to normal workspace usage.

- ``config/build/no_run.yaml`` — scripts to skip during automated runs. Each entry is a filename stem
  or path pattern with an inline comment explaining why it is skipped.
- ``config/build/env_vars.yaml`` — environment variables applied to each script during automated runs.
  Defines default values (e.g. test mode, small datasets) and per-script overrides for scripts
  that need different settings.
