# All necessary libraries and imports from other files.
from biomeld.ContextManager import ContextManager
from biomeld.Evaluator import Evaluator
from lxml import etree
import subprocess


# This class extends from the Evaluator class and are focused on the evaluation of BHM morhologies.
class HyperNEATEvaluator(Evaluator):

    # Constructor
    def __init__(self, evaluator):

        super().__init__(evaluator)
        self.number_of_soft_voxels = 0
        self.number_of_no_voxels = 0

    # This method receives the data necessary to populate the .vxa file, which interacts with the Voxelyze instance.
    def create_locomotion_file(self, individual_data):

        self.set_layers(individual_data.get("layers"))
        self.set_offsets(individual_data.get("offsets"))

        with open(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/individual_file.vxa", 'wb') as f:

            self.raw_tree.write(f, encoding="ISO-8859-1", pretty_print=True)

    # This method evaluates the BHM morphology through a Voxelyze instance. It returns the maximum displacement and the
    # number of voxels, including the type of voxels.
    def evaluate_individual(self):

        aptitude = {"voxels": {"total": -1}, "displacement": None}

        if self.check_morphology_consistency():

            with ContextManager(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"):

                command = self.VOXELYZE_COMMAND + "individual_file.vxa"
                subprocess.run(command, shell=True)

            with ContextManager(self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"):

                aptitude_tree = etree.parse(self.FITNESS_FILE, self.parser)
                root = aptitude_tree.getroot()
                aptitude["voxels"] = {"total": root.find("Fitness").find("VoxelNumber").text}
                aptitude["displacement"] = root.find("Fitness").find("normAbsoluteDisplacement").text

            self.__count_number_of_voxels()
            aptitude["voxels"]["hard"] = self.number_of_hard_voxels
            aptitude["voxels"]["soft"] = self.number_of_soft_voxels
            aptitude["voxels"]["absence"] = self.number_of_no_voxels

        self.__clean_data_and_offsets()

        return aptitude

    # This method counts the number of voxels that compose a BHM morphology.
    def __count_number_of_voxels(self):

        for layer in self.root.find("VXC").find("Structure").find("Data"):

            for number in layer.text:

                if number == "0":

                    self.number_of_no_voxels += 1
                    continue

                if number == "1":

                    self.number_of_hard_voxels += 1
                    continue

                if number == "3":

                    self.number_of_soft_voxels += 1
                    continue

    # This method sets the number of voxels and delete the data related to the BHM morphology.
    def __clean_data_and_offsets(self):

        self.clean_data()
        self.number_of_hard_voxels = 0
        self.number_of_soft_voxels = 0
        self.number_of_no_voxels = 0

