# All necessary libraries and imports from other files.
from locomotion.client.neat.NEATAlgorithmL import NEATAlgorithmL
import time

if __name__ == '__main__':

    print("************ BioMeld domain implementing NEAT algorithm. ************")

    # If needed, don't forget to update the path of the configuration file!
    configuration_path = "locomotion_NEAT.cfg"

    # Configuration of the CPPNs.
    cppn_data = {

        "input_neurons": 3,
        "output_neurons": 2,
        "recurrent_topology": True
    }

    # Configuration of the GA.
    ga_data = {

        "number_of_individuals": 10,
        "number_of_generations": 2,
        "reproduction_ratio": 0.4,
        "individuals_in_elitism": 2
    }

    # Configuration of the fitness function and the voxel layout.
    fitness_metrics = {

        "threshold": 0.999,
        "layout": {"x": 8, "y": 8, "z": 7},
        "displacement": 30.0
    }

    # Configuration of the server.
    server_configuration = {

        # This number should be the same as the number of Voxelyze instances deployed in the server side.
        "simulator_instances": 18,
        "initial_port": 8081,
        "target_ip_address": "http://192.168.0.89:"  # This is the server's ip.
    }

    for exp_id in range(0, 1):

        neat = NEATAlgorithmL(configuration_path, cppn_data, ga_data, fitness_metrics, server_configuration)
        start = time.time()
        neat.execute_algorithm(exp_id)
        end = time.time()
        print("Elapsed time: " + str(end - start))

        path = "fittest/time_" + str(exp_id)

        with open(path, "w") as file:

            file.write(str(end - start))
