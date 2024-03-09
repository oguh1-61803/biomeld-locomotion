# This class represents a neuron. It is used to compose a substrate. It receives the neuron id, the coordinates of the
# neuron, and the bias value of the neuron.
class Neuron:

    COORDINATES_INDEX = 0
    WEIGHT_INDEX = 1

    # Constructor
    def __init__(self, n_id, coo, bi=0.0):

        self._neuron_id = n_id
        self._coordinates = coo
        self._bias = bi
        self._links = None

    @property
    def neuron_id(self):

        return self._neuron_id

    @property
    def coordinates(self):

        return self._coordinates

    @property
    def bias(self):

        return self._bias

    @bias.setter
    def bias(self, b):

        self._bias = b

    @property
    def links(self):

        return self._links

    @links.setter
    def links(self, li):

        self._links = {}
        self._links = li

    # This function sets the bias to 0.0.
    def reset_bias(self):

        self._bias = 0.0

    # This function sets the bias and weights to 0.0.
    def reset_weights_and_bias(self):

        self._bias = 0.0

        for values in self._links.values():

            values[self.WEIGHT_INDEX] = 0.0

    # This function set the weight of a connection using the neuron id.
    def set_weight(self, neuron_id, weight):

        link = self._links.get(neuron_id)
        link[self.WEIGHT_INDEX] = weight
