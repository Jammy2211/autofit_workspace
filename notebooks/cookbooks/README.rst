The ``cookbooks`` folder contains cookbooks concisely explaining how to use different aspects of **PyAutoFit**.

Files
-----

- ``model.py``: Using the model composition API (e.g. `af.Model()`, `af.Collection()`).
- ``analysis.py``: Defining an`Analysis` class (e.g. define a `log_likelihood_function`, output visualization).
- ``search.py``: Custom settings of non-linear searches and a list of all searches available (e.g. MCM, nested sampling).
- ``results.py``: Using results from a non-linear search (e.g. `Samples` objects, maximum likelihood model, parameter errors).
- ``config.py``: Defining configuration files associated with your user define model (e.g. automatic prior setup and parameter labels).
- ``multiple_datasets.py``: Fitting multiple datasets simultaneously via `Analysis` class summing.
- ``database.py``: Using an sqlite3 database to store results of non-linear searches for fits to large datasets.
- ``multi_level_model.py``: Composing multi-level models from hierarchicies of Python classes.