# All necessary libraries and imports from other files.
from test_functions.hyperneat.HyperNEATAlgorithmTF import HyperNEATAlgorithmTF

if __name__ == '__main__':

    '''
            There are 5 functions implemented. To call each, pass the parameter in 'method get_function_data' as follows:
            - Gramacy & Lee: 'gl'
            - Rastrigin: 'r'
            - Levy 13: 'l13'
            - Ackley: 'a'
            - Bukin 6: 'b'
        '''

    print("************ Test functions with HyperNEAT algorithm. ************")

    # If needed, don't forget to update the path of the configuration file!
    configuration_path = "test_functions_HyperNEAT.cfg"

    # Configuration of the substrate.
    substrate = {

        "hidden_ranges": [(-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0)],
    }

    # Configuration of the CPPNs.
    cppn_data = {

        "input_neurons": 4,
        "output_neurons": 1,
        "recurrent_topology": True
    }

    # Configuration of the GA.
    ga_data = {

        "number_of_individuals": 100,
        "number_of_generations": 200,
        "individuals_in_elitism": 1,
        "fitness_threshold": 0.90
    }

    function_name = "gl"

    for _ in range(0, 1):

        functions_hyperneat = HyperNEATAlgorithmTF(configuration_path, substrate, cppn_data, ga_data, function_name)
        functions_hyperneat.execute_algorithm()
