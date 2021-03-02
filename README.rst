PyAutoFit Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02550/status.svg
   :target: https://doi.org/10.21105/joss.02550

|binder| |JOSS|

`Installation Guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautofit.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/26262bc184d0c77795db70636a004c9dce9c52b0?filepath=introduction.ipynb>`_ |
`HowToFit <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_

Welcome to the **PyAutoFit** Workspace. You can get started right away by going to the `autofit workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/26262bc184d0c77795db70636a004c9dce9c52b0?filepath=introduction.ipynb>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautofit.readthedocs.io>`_.

Getting Started
---------------

If you haven't already, install `PyAutoFit via pip or conda <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_.

Next, clone the ``autofit workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autofit_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Workspace Structure
===================

The workspace includes the following main directories:

- ``notebooks`` - **PyAutoFit** examples written as Jupyter notebooks.
- ``scipts`` - **PyAutoFit** examples written as Python scripts.
- ``config`` - Configuration files which customize **PyAutoFit**'s behaviour.
- ``dataset`` - Where data is stored, including example datasets distributed with **PyAutoFit**.
- ``output`` - Where the **PyAutoFit** analysis and visualization are output.

The examples in the notebooks and scripts folders are structured as follows:

- ``overview`` - Examples using **PyAutoFit** to compose and fit a model to data via a non-linear search.
- ``howtofit`` - Detailed step-by-step tutorials
- ``features`` - Examples of **PyAutoFit**'s advanced modeling features.

Getting Started
===============

We recommend new users begin with the example notebooks / scripts in the *overview* folder and the **HowToFit**
tutorials.

Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoFit v0.73.1**.

Support
=======

Support for installation issues and integrating your modeling software with **PyAutoFit** is available by
`raising an issue on the autofit_workspace GitHub page <https://github.com/Jammy2211/autofit_workspace/issues>`_. or
joining the **PyAutoFit** `Slack channel <https://pyautofit.slack.com/>`_, where we also provide the latest updates on
**PyAutoFit**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
