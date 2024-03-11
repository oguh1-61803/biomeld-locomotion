# All necessary libraries and imports from other files.
from locomotion.client.hyperneat.HyperNEATEvaluator import HyperNEATEvaluator
from locomotion.client.ActivationFunctionBank import ActivationFunctionBank
from configupdater import ConfigUpdater
import neat


# This is the main class. It orchestrates the execution of HyperNEAT. It receives the path of the HyperNEAT configuration
# file. It also receives the configuration of: the substrate, CPPNs, the GA, the fitness function, the server data.
class HyperNEATAlgorithmL:

    # Constructor
    def __init__(self, path, substrate_data, cppn_data, ga_data, fitness_metrics, server_configuration):

        self.activation_function_bank = ActivationFunctionBank()
        self.configuration_file_path = path
        self.number_of_generations = ga_data.get("number_of_generations")
        self.fitness_function = None
        self.preview = ga_data.get("preview")

        self.__configure_file(cppn_data, ga_data, fitness_metrics)
        self.__configure_fitness_function(substrate_data, fitness_metrics, server_configuration, cppn_data, ga_data.get("number_of_individuals"))

    # This is the main function; it triggers the mechanism of HyperNEAT.
    def execute_algorithm(self, experiment_id):

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

        fittest_substrate_network = self.fitness_function.parallel_build_substrate(fittest_cppn)

        self.fitness_function.get_fittest_individual_file(fittest_substrate_network, experiment_id)

    # This function configures the input file for HyperNEAT algorithm.
    def __configure_file(self, cppn_data, ga_data, fitness_metrics):

        updater = ConfigUpdater()
        updater.read(self.configuration_file_path)
        updater["NEAT"]["pop_size"].value = ga_data.get("number_of_individuals")
        updater["NEAT"]["fitness_threshold"].value = fitness_metrics.get("threshold")
        updater["DefaultReproduction"]["elitism"].value = ga_data.get("individuals_in_elitism")
        updater["DefaultGenome"]["num_inputs"].value = cppn_data.get("input_neurons")
        updater["DefaultGenome"]["num_outputs"].value = cppn_data.get("output_neurons")
        updater["DefaultReproduction"]["survival_threshold"].value = ga_data.get("reproduction_ratio")
        updater.update_file()

    # This function initialises the fitness function.
    def __configure_fitness_function(self, substrate_data, fitness_data, server_data, cppn_data, number_of_individuals):

        self.fitness_function = HyperNEATEvaluator(substrate_data, fitness_data, server_data, cppn_data, number_of_individuals)