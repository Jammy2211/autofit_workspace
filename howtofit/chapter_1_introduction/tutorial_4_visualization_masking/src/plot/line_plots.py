import matplotlib.pyplot as plt

# To visualize images during a phase, we need to be able to output them to hard-disk as a file (e.g a .png'). The line
# plot function below has been extended to provide this functionality.


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
