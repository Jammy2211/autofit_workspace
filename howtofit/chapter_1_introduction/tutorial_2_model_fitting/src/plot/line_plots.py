import matplotlib.pyplot as plt


def line(xvalues, line, ylabel=None):

    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()
