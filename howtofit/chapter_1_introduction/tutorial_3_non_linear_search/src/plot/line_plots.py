import matplotlib.pyplot as plt

# The 'line_plots.py' module is unchanged from the previous tutorial.


def line(xvalues, line, ylabel=None):

    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()
