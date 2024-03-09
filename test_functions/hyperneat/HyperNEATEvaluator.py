# All necessary libraries and imports from other files.
from test_functions.TestFunctionBank import TestFunctionBank
from test_functions.hyperneat.Substrate import Substrate
import math
import neat


# This class defines the fitness function. It evaluates individuals (CPPNs) utilising the mean squared error (MSE). It
# receives the nickname of the function, the configuration of the substrate, and if CPPNs are recurrent or not.
class HyperNEATEvaluator:

    X_BIAS = Y_BIAS = 0.0

    # Constructor
    def __init__(self, name, substrate_data, recurrent):

        self.function_name = name
        self.is_recurrent = recurrent
        self.functions_bank = TestFunctionBank()
        self.substrate = Substrate()
        self.substrate.initialise_substrate(substrate_data, self.function_name)

    # This function evaluates the population of CPPNs through the MSE.
    def evaluate_individuals(self, genomes, configuration):

        function_data = self.functions_bank.get_function_data(self.function_name)

        for genome_id, genome in genomes:

            if self.is_recurrent:

                cppn = neat.nn.RecurrentNetwork.create(genome, configuration)

            else:

                cppn = neat.nn.FeedForwardNetwork.create(genome, configuration)

            self.__build_substrate(cppn)
            neural_network = self.__build_neural_network_from_substrate()

            if self.function_name == "gl" or self.function_name == "r":

                x_data = function_data.get("x")
                y_data = function_data.get("y")
                y_predicted = []

                for x in x_data:

                    output = neural_network.activate([x])
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

                    output = neural_network.activate([xy[0], xy[1]])
                    z_predicted.append(output.pop())

                error = 0.0

                for z_d, z_p in zip(z_data, z_predicted):

                    difference = math.pow(z_d - z_p, 2)

                    if difference >= 1.0:

                        error += 1.0

                    else:

                        error += difference

                genome.fitness = 1.0 - (error / len(z_data))

    # This function generates a substrate by querying a CCPN.
    def __build_substrate(self, cppn):

        for layer in self.substrate.hidden_layers:

            for neuron in layer:

                coordinates_b = neuron.coordinates

                for key in neuron.links.keys():

                    link = neuron.links.get(key)
                    coordinates_a = link[0]
                    weight_cppn_input = (coordinates_a[0], coordinates_a[1], coordinates_b[0], coordinates_b[1])
                    link[1] = cppn.activate(weight_cppn_input).pop()

                bias_input = (coordinates_b[0], coordinates_b[0], self.X_BIAS, self.Y_BIAS)
                neuron.bias = cppn.activate(bias_input).pop()

        for output_neuron in self.substrate.output_layer:

            coordinates_b = output_neuron.coordinates

            for key in output_neuron.links.keys():

                link = output_neuron.links.get(key)
                coordinates_a = link[0]
                weight_cppn_input = (coordinates_a[0], coordinates_a[1], coordinates_b[0], coordinates_b[1])
                link[1] = cppn.activate(weight_cppn_input).pop()

            bias_input = (coordinates_b[0], coordinates_b[0], self.X_BIAS, self.Y_BIAS)
            output_neuron.bias = cppn.activate(bias_input).pop()

    # This method generates a substrate using the fittest CPPN found by HyperNEAT.
    def build_fittest_substrate(self, cppn):

        self.__build_substrate(cppn)
        fittest_substrate = self.__build_neural_network_from_substrate()

        return fittest_substrate

    # This method returns a functional neural network from a substrate.
    def __build_neural_network_from_substrate(self):

        input_neurons = self.substrate.get_input_neurons()
        output_neurons = self.substrate.get_output_neurons()
        neurons_evals = self.substrate.get_neurons_evals()
        substrate_net = neat.nn.RecurrentNetwork(input_neurons, output_neurons, neurons_evals)

        return substrate_net
