# All necessary libraries and imports from other files.
from test_functions.ActivationFunctionBank import ActivationFunctionBank
from test_functions.neat.NEATEvaluator import NEATEvaluator
from configupdater import ConfigUpdater
import neat


# This is the main class. It orchestrates the execution of NEAT. It receives the path of the NEAT configuration file.
# It also receives the configuration of: CPPNs, the GA, and the nickname of the function.
class NEATAlgorithmTF:

    # Constructor
    def __init__(self, path, cppn_data, ga_data, function_name):

        self.activation_function_bank = ActivationFunctionBank()
        self.configuration_file_path = path
        self.number_of_generations = ga_data.get("number_of_generations")
        self.fitness_function = None
        self.function_name = function_name
        self.__configure_file(cppn_data, ga_data)
        self.__configure_fitness_function(cppn_data.get("recurrent_topology"))

    # This is the main function; it triggers the mechanism of NEAT.
    def execute_algorithm(self):

        configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, self.configuration_file_path)
        population = neat.Population(configuration)
        reporter = neat.StdOutReporter(True)
        stats = neat.StatisticsReporter()
        population.add_reporter(reporter)
        population.add_reporter(stats)

        configuration.genome_config.add_activation("neg_abs", self.activation_function_bank.negative_abs)
        configuration.genome_config.add_activation("neg_square", self.activation_function_bank.negative_square)
        configuration.genome_config.add_activation("sqrt_abs", self.activation_function_bank.square_abs)
        configuration.genome_config.add_activation("neg_sqrt_abs", self.activation_function_bank.negative_square_abs)
        configuration.genome_config.add_activation("neg_sin", self.activation_function_bank.negative_sin)
        configuration.genome_config.add_activation("cos", self.activation_function_bank.cos)
        configuration.genome_config.add_activation("neg_cos", self.activation_function_bank.negative_cos)
        configuration.genome_config.add_activation("gelu", self.activation_function_bank.gelu)

        fittest = population.run(self.fitness_function.evaluate_individuals, self.number_of_generations)
        print("*-" * 97)
        print("Best individual:")
        print("{!s}".format(fittest))
        print("%s nodes with %s connections." % (fittest.size()[0], fittest.size()[1]))
        print("*-" * 97)

        if self.fitness_function.is_recurrent:

            fittest_cppn = neat.nn.RecurrentNetwork.create(fittest, configuration)

        else:

            fittest_cppn = neat.nn.FeedForwardNetwork.create(fittest, configuration)

        data = self.fitness_function.functions_bank.get_function_data(self.function_name)

        if self.function_name == "gl" or self.function_name == "r":

            for x, y in zip(data.get("x"), data.get("y")):

                output = fittest_cppn.activate([x])
                print("Input: {!r}; expected output: {!r}; obtained: {!r}".format(x, y, output))

        else:

            for xy, z in zip(data.get("xy"), data.get("z")):

                output = fittest_cppn.activate([xy[0], xy[1]])
                print("Input: {!r}; expected output: {!r}; obtained: {!r}".format(xy, z, output))

    # This function configures the input file for NEAT algorithm.
    def __configure_file(self, cppn_data, ga_data):

        updater = ConfigUpdater()
        updater.read(self.configuration_file_path)
        updater["NEAT"]["pop_size"].value = ga_data.get("number_of_individuals")
        updater["NEAT"]["fitness_threshold"].value = ga_data.get("fitness_threshold")
        updater["DefaultReproduction"]["elitism"].value = ga_data.get("individuals_in_elitism")
        updater["DefaultGenome"]["num_outputs"].value = cppn_data.get("output_neurons")

        if self.function_name == "gl" or self.function_name == "r":

            updater["DefaultGenome"]["num_inputs"].value = 1

        else:

            updater["DefaultGenome"]["num_inputs"].value = 2

        updater.update_file()

    # This function initialises the fitness function.
    def __configure_fitness_function(self, recurrent):

        self.fitness_function = NEATEvaluator(self.function_name, recurrent)
