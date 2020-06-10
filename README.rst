PyAutoFit Workspace
====================

Welcome to the **PyAutoFit** Workspace. If you haven't already, you should install **PyAutoFit**, following the
instructions at `the PyAutoFit readthedocs <https://pyautofit.readthedocs.io/en/master/installation.html>`_.

Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoFit v0.57.2**.

.. code-block:: python

    pip install autofit==0.57.2

Getting Started
===============

To begin, check out the 'api/simple' folder. This will explain the best way to get started with **PyAutoFit**.

Workspace Contents
==================

The workspace includes the following:

- **API** - Illustrative scripts of the **PyAutoFit** interface, for examples on how to make a model, fit it to data,
            etc.
- **Config** - Configuration files which customize **PyAutoFit**'s behaviour.
- **Dataset** - Where data is stored, including example datasets distributed with **PyAutoFit**.
- **HowToFit** - The **HowToFit** lecture series.
- **Output** - Where the **PyAutoFit** analysis and visualization are output.

Issues, Help and Problems
=========================

If the installation below does not work or you have issues running scripts in the workspace, please post an issue on
the `issues section of the autofit_workspace <https://github.com/Jammy2211/autofit_workspace/issues>`_.

Setup
=====

The workspace is independent from the autofit install (e.g. the 'site-packages' folder), meaning you can edit
workspace scripts and not worry about conflicts with new **PyAutoFit** installs.

Python therefore must know where the workspace is located so that it can import modules / scripts. This is done by 
setting the PYTHONPATH:

.. code-block:: python

    export PYTHONPATH=/path/to/autofit_workspace/

**PyAutoFit** additionally needs to know the default location of config files, which is done by setting the WORKSPACE.
Clone autofit workspace & set WORKSPACE enviroment variable:

.. code-block:: python

    export WORKSPACE=/path/to/autofit_workspace/

Matplotlib uses the default backend on your computer, as set in the config file 
autofit_workspace/config/visualize/general.ini:
 
.. code-block:: python

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the 
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: python

    [general]
    backend = TKAgg

You can test everything is working by running the example pipeline runner in the autofit_workspace

.. code-block:: python

    python3 /path/to/autofit_workspace/api/simple/fit.py

Support & Discussion
====================

If you haven't already, go ahead and `email <https://github.com/Jammy2211>`_ me to get on our
`Slack channel <https://pyautofit.slack.com/>`_.
