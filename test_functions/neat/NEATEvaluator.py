# All necessary libraries and imports from other files.
from src.test_functions.TestFunctionBank import TestFunctionBank
import math
import neat

# This class defines the fitness function. It evaluates individuals (CPPNs) utilising the mean squared error (MSE). It
# receives the nickname of the function and if CPPNs are recurrent or not.
class NEATEvaluator:

    # Constructor
    def __init__(self, name, recurrent):

        self.function_name = name
        self.is_recurrent = recurrent
        self.functions_bank = TestFunctionBank()

    # This function evaluates the population of CPPNs through the MSE.
    def evaluate_individuals(self, genomes, configuration):

        function_data = self.functions_bank.get_function_data(self.function_name)

        for genome_id, genome in genomes:

            if self.is_recurrent:

                cppn = neat.nn.RecurrentNetwork.create(genome, configuration)

            else:

                cppn = neat.nn.FeedForwardNetwork.create(genome, configuration)

            if self.function_name == "gl" or self.function_name == "r":

                x_data = function_data.get("x")
                y_data = function_data.get("y")
                y_predicted = []

                for x in x_data:

                    output = cppn.activate([x])
                    y_predicted.append(output.pop())

                error = 0.0

                for y_d, y_p in zip(y_data, y_predicted):

                    difference = math.pow(y_d - y_p, 2)

                    if difference >= 1.0:

                        error += 1.0

                    else:

                        error += difference

                genome.fitness = 1.0 - (error / len(y_data))

            else:

                xy_data = function_data.get("xy")
                z_data = function_data.get("z")
                z_predicted = []

                for xy in xy_data:

                    output = cppn.activate([xy[0], xy[1]])
                    z_predicted.append(output.pop())

                error = 0.0

                for z_d, z_p in zip(z_data, z_predicted):

                    difference = math.pow(z_d - z_p, 2)

                    if difference >= 1.0:

                        error += 1.0

                    else:

                        error += difference

                genome.fitness = 1.0 - (error / len(z_data))
