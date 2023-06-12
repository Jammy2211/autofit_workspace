class LinearFit:
    def __init__(self, gradient=1.0, intercept=0.0):
        self.gradient = gradient
        self.intercept = intercept

    def line_from_xvalues(self, xvalues):
        return self.gradient * xvalues + self.intercept


class PowerFit:
    def __init__(self, gradient=1.0, intercept=0.0, power=1.0):
        self.gradient = gradient
        self.intercept = intercept
        self.power = power

    def line_from_xvalues(self, xvalues):
        return self.gradient * ((xvalues) ** self.power) + self.intercept
