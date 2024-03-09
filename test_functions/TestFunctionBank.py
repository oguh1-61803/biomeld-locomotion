# All necessary libraries.
import math

# This class works as a bank of functions. It receives a string representing the name of the mathematical function and
# returns the data associated with the function in a determined range.
class TestFunctionBank:

    # Constructor
    def __init__(self):

        self.data = {}

    # This function receives the function nickname and returns a tuple containing data associated with the
    # independent and dependent variable(s).
    def get_function_data(self, function_name):

        self.data = {}

        if function_name == "gl":

            return self.__generate_gramacy_lee_data()

        elif function_name == "r":

            return self.__generate_rastrigin_data()

        elif function_name == "l13":

            return self.__generate_levy_13_data()

        elif function_name == "a":

            return self.__generate_ackley_function()

        elif function_name == "b":

            return self.__generate_bukin_6_data()

    # Function that generates data of the Gramacy & Lee mathematical function.
    def __generate_gramacy_lee_data(self):

        x_values = []
        y_values = []
        x = 0.5

        while x <= 2.5:

            x_values.append(x)
            x += 0.05
            x = round(x, 2)

        for x_value in x_values:

            term_1 = math.sin((10 * math.pi * x_value)) / (x_value * 2)
            term_2 = math.pow(x_value - 1, 4)
            y = round(term_1 + term_2, 5)
            y_values.append(y)

        self.data["x"] = x_values
        self.data["y"] = y_values

        return self.data

    # Function that generates data of the Rastrigin mathematical function.
    def __generate_rastrigin_data(self):

        x_values = []
        y_values = []
        x = -5.12

        while x <= 5.12:

            x_values.append(x)
            x += 0.32
            x = round(x, 2)

        for x_value in x_values:

            term_2 = math.pow(x_value, 2) - (10 * math.cos(2 * math.pi * x_value))
            y = round(10 + term_2, 5)
            y_values.append(y)

        self.data["x"] = x_values
        self.data["y"] = y_values

        return self.data

    # Function that generates data of the Levy No. 13 mathematical function.
    def __generate_levy_13_data(self):

        x_ref = []
        y_ref = []
        ref = -10

        while ref <= 10:

            x_ref.append(ref)
            y_ref.append(ref)
            ref += 1.0
            ref = round(ref, 1)

        xy_values = []
        z_values = []

        for x_value in x_ref:

            for y_value in y_ref:

                term_1 = math.pow(math.sin(3 * math.pi * x_value), 2)
                term_2 = math.pow(x_value - 1, 2) * (1 + math.pow(math.sin(3 * math.pi * y_value), 2))
                term_3 = math.pow(y_value - 1, 2) * (1 + math.pow(math.sin(2 * math.pi * y_value), 2))
                z = round(term_1 + term_2 + term_3, 5)
                z_values.append(z)
                xy_values.append([x_value, y_value])

        self.data["xy"] = xy_values
        self.data["z"] = z_values

        return self.data

    # Function that generates data of the Ackley mathematical function.
    def __generate_ackley_function(self):

        x_ref = []
        y_ref = []
        ref = -5

        while ref <= 5:

            x_ref.append(ref)
            y_ref.append(ref)
            ref += 1.0
            ref = round(ref, 1)

        xy_values = []
        z_values = []

        for x_value in x_ref:

            for y_value in y_ref:

                term_1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (math.pow(x_value, 2) + math.pow(y_value, 2))))
                term_2 = -math.exp(0.5 * (math.cos(2 * math.pi * x_value) + math.cos(2 * math.pi * y_value)))
                z = round(term_1 + term_2 + math.e + 20, 5)
                z_values.append(z)
                xy_values.append([x_value, y_value])

        self.data["xy"] = xy_values
        self.data["z"] = z_values

        return self.data

    # Function that generates data of the Bukin No.6 mathematical function.
    def __generate_bukin_6_data(self):

        x_ref = []
        ref = -15

        while ref <= -5:

            x_ref.append(ref)
            ref += 1.0
            ref = round(ref, 1)

        y_ref = []
        ref = -3

        while ref <= 3:

            y_ref.append(ref)
            ref += 1.0
            ref = round(ref, 1)

        xy_values = []
        z_values = []

        for x_value in x_ref:

            for y_value in y_ref:

                term_1 = 100 * math.sqrt(math.fabs(y_value - (0.01 * math.pow(x_value, 2))))
                term_2 = 0.01 * math.fabs(x_value + 10)
                z = round(term_1 + term_2, 5)
                z_values.append(z)
                xy_values.append([x_value, y_value])

        self.data["xy"] = xy_values
        self.data["z"] = z_values

        return self.data
