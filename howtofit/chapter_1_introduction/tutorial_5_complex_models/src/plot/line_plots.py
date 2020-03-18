import matplotlib.pyplot as plt

# This module is unchanged from the previous tutorial.


def line(
    xvalues,
    line,
    ylabel=None,
    output_path=None,
    output_filename=None,
    output_format="show",
):

    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(output_path + output_filename + ".png")
    plt.clf()
