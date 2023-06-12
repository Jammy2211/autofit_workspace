The ``features`` folder contains example scripts for using advanced **PyAutoFit** features.

Files (Advanced)
----------------

- ``database.py``: Writing and loading results to a sqlite3 database.
- ``multiple_datasets.py``: Defining multiple ``Analysis`` classes (and therefore ``log_likelihood_function()``'s) and summing them to perform a joint fit.
- ``search_chaining.py``: Chaining together multiple non-linear searches to automate the fitting of a complex model.
- ``search_grid_search.py``: Perform a massively parallel grid search of non-linear searches.
- ``sensitivity_mapping.py``: Determine how sensitive a dataset is to a model's complexity.