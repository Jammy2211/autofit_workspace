The ``optimize`` folder contains example scripts which fit a model using the different optimizers supported by **PyAutoFit**:

Files
-----

- ``drawer.py``: Fit a model using the in-built ``Drawer`` which simple draws samples from the priors.
- ``LBFGS.py``: Fit a model using the scipy lbfgs optimizer.
- ``PySwarmsGlobal.py``: Fit a model using the optimizer algorithm PySwarms, using its global fitting method.
- ``PySwarmsLocal.py``: Fit a model using the optimizer algorithm PySwarms, using its local fitting method.

- ``analysis.py``: Not an example, an ``Analysis`` class used in the examples.
- ``model.py``: Not an example, defines the model component used in the examples.