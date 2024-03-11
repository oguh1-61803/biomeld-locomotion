# All necessary libraries and imports from other files.
from concurrent.futures import ProcessPoolExecutor
import numpy
import requests
import base64
import random
import neat


# This class defines the fitness function. It evaluates individuals (morphologies) through Voxelyze instances which are
# deployed in the server. It receives the configuration data of: fitness function, server, CPNNs, and the number of
# individuals.
class NEATEvaluator:

    FITTEST_PATH = "fittest/"
    MAPPING_REFERENCE = 0.5

    # Constructor
    def __init__(self, fitness_data, server_data, cppn_data, number_of_individuals):

        self.x_vector = None
        self.y_vector = None
        self.z_vector = None
        self.middle = None
        self.quarter = None
        self.quarter = None
        self.middle_plus_quarter = None
        self.middle_minus_quarter = None
        self.offsets = []
        self.target_url = None
        self.number_of_workers = None
        self.is_recurrent = cppn_data.get("is_recurrent")
        self.fitness_data = fitness_data
        self.number_of_voxels = None

        self.__initialise_layout(number_of_individuals)
        self.__initialise_server(server_data)

    # This function evaluates the population of morphologies through the Voxelyze instances deployed in the server.
    def evaluate_individuals(self, genomes, config):

        list_of_cppns = []
        list_of_robot_morphologies = []

        for genome_id, genome in genomes:

            if self.is_recurrent:

                cppn = neat.nn.RecurrentNetwork.create(genome, config)
                list_of_cppns.append(cppn)

            else:

                cppn = neat.nn.FeedForwardNetwork.create(genome, config)
                list_of_cppns.append(cppn)

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for morphology in executor.map(self.parallel_build_robot_morphology, list_of_cppns, chunksize=2):

                list_of_robot_morphologies.append(morphology)

        list_of_fitness_values = []
        random.shuffle(self.offsets)

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for fitness_value in executor.map(self.parallel_evaluate_morphology_in_server, list_of_robot_morphologies, self.offsets, chunksize=2):

                list_of_fitness_values.append(fitness_value)

        if len(genomes) > len(list_of_fitness_values):

            print(len(genomes), len(list_of_fitness_values))
            missed_aptitudes = len(genomes) - len(list_of_fitness_values)

            for _ in range(0, missed_aptitudes):

                list_of_fitness_values.append(random.uniform(0.25, 0.50))

        index = 0

        for genome_id, genome in genomes:

            genome.fitness = list_of_fitness_values[index]
            index += 1

    # This function returns the morphology built by a CPPN, which is received as a parameter.
    def parallel_build_robot_morphology(self, cppn):

        morphology = []

        for z_coordinate in self.z_vector:

            layer = ""

            for y_coordinate in self.y_vector:

                for x_coordinate in self.x_vector:

                    cpp_input = [x_coordinate, y_coordinate, z_coordinate]
                    cppn_output = cppn.activate(cpp_input)

                    vp = numpy.fabs(cppn_output[0])

                    if vp < self.MAPPING_REFERENCE:

                        layer += "0"

                    else:

                        m = numpy.fabs(cppn_output[1])

                        if m < self.MAPPING_REFERENCE:

                            layer += "1"

                        else:

                            layer += "3"

            morphology.append(layer)

        return morphology

    # This function sends a morphology (and an offset) to the server to be evaluated and returns the evaluation
    # provided by the server.
    def parallel_evaluate_morphology_in_server(self, morphology, offset):

        robot = {

            "evaluation": True,
            "layers": morphology,
            "offsets": offset
        }

        r = requests.get(url=self.target_url + "/biomeld-hn", json=robot).json()
        print(r)

        if float(r.get("voxels").get("total")) == -1.0 or float(r.get("voxels").get("soft")) == 0:

            return 0.0

        displacement_aptitude = float(r.get("displacement")) / self.fitness_data.get("displacement")

        if (self.middle - self.quarter) <= int(r.get("voxels").get("total")) <= (self.middle + self.quarter):

            minimal_voxels_aptitude = 1.0

        elif (self.middle_minus_quarter - self.eighth) <= int(r.get("voxels").get("total")) < self.middle_minus_quarter:

            minimal_voxels_aptitude = 0.3333

        elif self.middle_plus_quarter < int(r.get("voxels").get("total")) <= (self.middle_plus_quarter + self.eighth):

            minimal_voxels_aptitude = 0.3333

        else:

            minimal_voxels_aptitude = 0.0

        return (displacement_aptitude * 0.5) + (minimal_voxels_aptitude * 0.5)

    # This function generates a .vxa file of the fittest morphology found by NEAT algorithm.
    def get_fittest_individual_file(self, cppn, experiment_id):

        morphology = self.parallel_build_robot_morphology(cppn)
        offset = random.choice(self.offsets)

        robot = {

            "evaluation": False,
            "layers": morphology,
            "offsets": offset
        }

        r = requests.get(url=self.target_url + "/biomeld-hn", json=robot).json()
        print(r)
        raw_file = r.get("individual_file")
        bytes_file = raw_file.encode()
        final_file = base64.b64decode(bytes_file)
        print(final_file)
        fitness_values = r.get("fitness_values")
        print("Number of voxels: ", fitness_values.get("voxels"))
        print("Displacement: ", fitness_values.get("displacement"))
        print("Instance used:", r.get("instance"))
        path = self.FITTEST_PATH + "fittest_" + str(experiment_id) + ".vxa"

        with open(path, "wb") as file:

            file.write(final_file)

    # This function initialises all the data necessary for the GA and the morphology files.
    def __initialise_layout(self, number_of_individuals):

        self.x_vector = [float(x) for x in range(0, self.fitness_data.get("layout").get("x"))]
        self.y_vector = [float(y) for y in range(0, self.fitness_data.get("layout").get("y"))]
        self.z_vector = [float(z) for z in range(0, self.fitness_data.get("layout").get("z"))]

        for _ in range(0, number_of_individuals):

            offset = self.__generate_offsets()
            self.offsets.append(offset)

        self.number_of_voxels = (self.fitness_data.get("layout").get("x") * self.fitness_data.get("layout").get("y") *
                                 self.fitness_data.get("layout").get("z"))

        self.middle = int(self.number_of_voxels / 2)
        self.quarter = int(self.number_of_voxels / 4)
        self.middle_plus_quarter = self.middle + self.quarter
        self.middle_minus_quarter = self.middle - self.quarter
        self.eighth = int(self.number_of_voxels / 8)

    # This function initialises the offsets, which are necessary to evaluate morphologies.
    def __generate_offsets(self):

        offsets = []

        for _ in range(0, len(self.z_vector)):

            offset = ""

            for _ in range(0, len(self.x_vector) * len(self.y_vector)):

                offset += str(random.uniform(-1.0, 1.0)) + ", "

            offsets.append(offset[:-2])

        return offsets

    # This function initialises the Voxelyze instances deployed in the server.
    def __initialise_server(self, server_config):

        self.target_url = server_config.get("target_ip_address") + "8000"
        self.number_of_workers = server_config.get("simulator_instances")
        initial_port = server_config.get("initial_port")

        for _ in range(0, self.number_of_workers):

            initialisation = {

                "voxels": (float(self.fitness_data.get("layout").get("x")),
                           float(self.fitness_data.get("layout").get("y")),
                           float(self.fitness_data.get("layout").get("z")))
            }

            print(initialisation)
            r = requests.post(url=server_config.get("target_ip_address") + str(initial_port) + "/biomeld-hn",
                              json=initialisation)
            print(r.json())
            initial_port += 1
