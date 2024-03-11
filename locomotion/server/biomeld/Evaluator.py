# All necessary libraries and imports from other files.
from biomeld.ContextManager import ContextManager
from lxml import etree
import subprocess
import re


# This abstract class provides the general behaviour of the morphology evaluator.
class Evaluator:

    X_VOXELS = "X_Voxels"
    Y_VOXELS = "Y_Voxels"
    Z_VOXELS = "Z_Voxels"

    # If needed, change the path of the root folder.
    PREFIX_PATH = "/home/BioMeld_Locomotion/biomeld/"
    VOXELYZE_PATH = PREFIX_PATH + "voxelyze/voxelyze"
    VOXELYZE_COMMAND = "./voxelyze -f "
    LOCOMOTION_PATH = PREFIX_PATH + "voxelyze/locomotion/base.vxa"
    EVALUATOR_PATH = PREFIX_PATH + "evaluators/"
    EVALUATOR_WORD = "evaluator_"
    FITNESS_FILE = "fitness.xml"

    INVALID_PATTERN = re.compile(r"\b(0)\1+\b")

    # Constructor
    def __init__(self, evaluator):

        self.parser = etree.XMLParser(remove_blank_text=True)
        self.raw_tree = etree.parse(self.LOCOMOTION_PATH, self.parser)
        self.root = self.raw_tree.getroot()
        self.number_of_hard_voxels = 0
        self.evaluator_id = str(evaluator)
        print(self.evaluator_id)

    # This method initialises the file of the morphologies that will be evaluated and generates a directory where one
    # Voxelyze is deployed.
    def initialise_xml_tree(self, conf):

        self.root.find("VXC").find("Structure").find(self.X_VOXELS).text = str(conf.get("voxels")[0])
        self.root.find("VXC").find("Structure").find(self.Y_VOXELS).text = str(conf.get("voxels")[1])
        self.root.find("VXC").find("Structure").find(self.Z_VOXELS).text = str(conf.get("voxels")[2])
        self.clean_data()
        self.root.find("Simulator").find("GA").find("FitnessFileName").text = "fitness.xml"

        with ContextManager(self.EVALUATOR_PATH):

            command = "mkdir " + self.EVALUATOR_WORD + self.evaluator_id + "/"
            subprocess.run(command, shell=True)

        command = "cp " + self.VOXELYZE_PATH + " " + self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/"
        subprocess.run(command, shell=True)

    def create_locomotion_file(self, individual_data):

        pass

    def evaluate_individual(self):

        pass

    # This function returns the path of the morphology file (a .vxa file). The path includes the number of instance that
    # evaluated the morphology
    def get_individual_file_path(self):

        return self.EVALUATOR_PATH + self.EVALUATOR_WORD + self.evaluator_id + "/individual_file.vxa"

    # This method removes the data of the morphology that was evaluated.
    def clean_data(self):

        for child in self.root.find("VXC").find("Structure").find("Data"):
            self.root.find("VXC").find("Structure").find("Data").remove(child)

        for child in self.root.find("VXC").find("Structure").find("PhaseOffset"):
            self.root.find("VXC").find("Structure").find("PhaseOffset").remove(child)

    # This method adds the data of the morphology that will be evaluated.
    def set_layers(self, individual_layers):

        for layer in individual_layers:

            l = etree.SubElement(self.root.find("VXC").find("Structure").find("Data"), "Layer")
            l.text = etree.CDATA(layer)

    # This method adds the phase offsets necessary for the simulation of morphologies.
    def set_offsets(self, individual_offsets):

        for offsets in individual_offsets:

            o = etree.SubElement(self.root.find("VXC").find("Structure").find("PhaseOffset"), "Layer")
            o.text = etree.CDATA(offsets)

    # This validates that the structure of the morphology has at least 1 voxel.
    def check_morphology_consistency(self):

        inconsistency_counter = 0
        consistency_reference = len(self.root.find("VXC").find("Structure").find("Data"))

        for layer in self.root.find("VXC").find("Structure").find("Data"):

            if re.search(self.INVALID_PATTERN, layer.text):

                inconsistency_counter += 1

        if inconsistency_counter == consistency_reference:

            return False

        else:

            return True

