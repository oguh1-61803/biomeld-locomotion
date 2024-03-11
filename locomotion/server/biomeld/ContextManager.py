# All necessary libraries.
import os


# This class helps to work with external resources. In this case, it helps to create folders and copy files. These
# operations are performed when the Voxelyze instances are initialised and during the evolutionary process.
class ContextManager:

    # Constructor
    def __init__(self, new_path):

        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):

        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):

        os.chdir(self.saved_path)
