from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.plot import (
    line_plots,
)

# The 'dataset_plots.py' module is unchanged from the previous tutorial.


def data(dataset):
    """Plot the data values of a Line dataset.

    Parameters
    -----------
    Line : dataset.Line
        The observed Line dataset whose data is plotted.
    """
    line_plots.line(xvalues=dataset.xvalues, line=dataset.data, ylabel="Data Values")


def noise_map(dataset):
    """Plot the noise map of a Line dataset.

    Parameters
    -----------
    Line : dataset.Line
        The observed Line dataset whose data is plotted.
    """
    line_plots.line(xvalues=dataset.xvalues, line=dataset.noise_map, ylabel="Noise-Map")
