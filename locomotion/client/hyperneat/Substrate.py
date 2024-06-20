# All necessary libraries and imports from other files.
from locomotion.client.hyperneat.Neuron import Neuron
from neat.activations import relu_activation
from neat.aggregations import sum_aggregation


# This class represents a substrate, which is a set of neurons arranged in a coordinates-based system.
class Substrate:

    RESPONSE = 1.0
    NO_CONNECTION_RANGE = (-0.2, 0.2)
    NORMALISATION_RANGE = (-3.0, 3.0)

    # Constructor
    def __init__(self):

        self.output_layer = []
        self.hidden_layers = []
        self.input_layer = []

    # This function inialises the substrate.
    def initialise_substrate(self, substrate_data):

        self.__initialise_neurons(substrate_data)
        self.__initialise_connections()

    # This function returns the input neurons.
    def get_input_neurons(self):

        input_neurons = []

        for neuron in self.input_layer:

            input_neurons.append(neuron.neuron_id)

        return input_neurons

    # This function returns the output neurons.
    def get_output_neurons(self):

        output_neurons = []

        for neuron in self.output_layer:

            output_neurons.append(neuron.neuron_id)

        return output_neurons

    # This function returns the topology of the substrate. In other words, it returns the data related to how the
    # neurons are connected and the weight values associated. The bias of neurons is also included.
    def get_neurons_evals(self):

        neurons_evals = []

        for layer in self.hidden_layers:

            for neuron in layer:

                if neuron.bias < self.NORMALISATION_RANGE[0]:

                    neuron.bias = self.NORMALISATION_RANGE[0]

                elif neuron.bias > self.NORMALISATION_RANGE[1]:

                    neuron.bias = self.NORMALISATION_RANGE[1]

                else:

                    raw_bias = abs(neuron.bias)
                    normalised_bias = raw_bias * self.NORMALISATION_RANGE[1]

                    if neuron.bias > 0.0:

                        neuron.bias = normalised_bias

                    elif neuron.bias < 0.0:

                        neuron.bias = normalised_bias * -1.0

                connections = []

                for k in neuron.links.keys():

                    link = neuron.links.get(k)

                    if self.NO_CONNECTION_RANGE[1] >= link[1] >= self.NO_CONNECTION_RANGE[0]:

                        continue

                    elif link[1] > 1.0:

                        connection = (k, self.NORMALISATION_RANGE[1])
                        connections.append(connection)

                    elif link[1] < -1.0:

                        connection = (k, self.NORMALISATION_RANGE[0])
                        connections.append(connection)

                    else:

                        raw_value = abs(link[1])
                        normalised_value = raw_value * self.NORMALISATION_RANGE[1]

                        if link[1] > 0.0:

                            connection = (k, normalised_value)
                            connections.append(connection)

                        elif link[1] < 0.0:

                            connection = (k, normalised_value * -1.0)
                            connections.append(connection)

                neuron_eval = (neuron.neuron_id, relu_activation, sum_aggregation, neuron.bias, self.RESPONSE, connections)
                neurons_evals.append(neuron_eval)

        for output_neuron in self.output_layer:

            if output_neuron.bias < self.NORMALISATION_RANGE[0]:

                output_neuron.bias = self.NORMALISATION_RANGE[0]

            elif output_neuron.bias > self.NORMALISATION_RANGE[1]:

                output_neuron.bias = self.NORMALISATION_RANGE[1]

            else:

                raw_bias = abs(output_neuron.bias)
                normalised_bias = raw_bias * self.NORMALISATION_RANGE[1]

                if output_neuron.bias > 0.0:

                    output_neuron.bias = normalised_bias

                elif output_neuron.bias < 0.0:

                    output_neuron.bias = normalised_bias * -1.0

            connections = []

            for k in output_neuron.links.keys():

                link = output_neuron.links.get(k)

                if self.NO_CONNECTION_RANGE[1] >= link[1] >= self.NO_CONNECTION_RANGE[0]:

                    continue

                elif link[1] > 1.0:

                    connection = (k, self.NORMALISATION_RANGE[1])
                    connections.append(connection)

                elif link[1] < -1.0:

                    connection = (k, self.NORMALISATION_RANGE[0])
                    connections.append(connection)

                else:

                    raw_value = abs(link[1])
                    normalised_value = raw_value * self.NORMALISATION_RANGE[1]

                    if link[1] > 0.0:

                        connection = (k, normalised_value)
                        connections.append(connection)

                    elif link[1] < 0.0:

                        connection = (k, normalised_value * -1.0)
                        connections.append(connection)

            neuron_eval = (output_neuron.neuron_id, relu_activation, sum_aggregation, output_neuron.bias, self.RESPONSE, connections)
            neurons_evals.append(neuron_eval)

        return neurons_evals

    # This function initialises the substrate.
    def __initialise_neurons(self, substrate_data):

        y_axis = -1.0

        n1 = Neuron(-1, (-0.5, y_axis))
        self.input_layer.append(n1)
        n2 = Neuron(-2, (0.0, y_axis))
        self.input_layer.append(n2)
        n3 = Neuron(-3, (0.5, y_axis))
        self.input_layer.append(n3)

        neuron_id = 0
        y_axis = len(substrate_data.get("hidden_ranges"))
        n4 = Neuron(neuron_id, (-0.5, y_axis))
        self.output_layer.append(n4)
        neuron_id += 1
        n5 = Neuron(neuron_id, (0.5, y_axis))
        self.output_layer.append(n5)
        neuron_id += 1

        y_axis = 0.0
        hidden_ranges = substrate_data.get("hidden_ranges")

        for hidden_range in hidden_ranges:

            x_axis = hidden_range[0]
            layer = []

            while x_axis != (hidden_range[1] + 1):

                n = Neuron(neuron_id, (x_axis, y_axis))
                layer.append(n)
                x_axis += 1.0
                neuron_id += 1

            self.hidden_layers.append(layer)
            y_axis += 1

    # This function initialises the connections among the neurons in the substrate.
    def __initialise_connections(self):

        for output_neuron in self.output_layer:

            links = {}

            for input_neuron in self.input_layer:

                link = [input_neuron.coordinates, 0.0]
                links[input_neuron.neuron_id] = link

            for hidden_layer in self.hidden_layers:

                for hidden_neuron in hidden_layer:

                    link = [hidden_neuron.coordinates, 0.0]
                    links[hidden_neuron.neuron_id] = link

            output_neuron.links = links

        for layer_index_a in reversed(range(len(self.hidden_layers))):

            for hidden_neuron_a in self.hidden_layers[layer_index_a]:

                links = {}

                for input_neuron in self.input_layer:

                    link = [input_neuron.coordinates, 0.0]
                    links[input_neuron.neuron_id] = link

                for layer_index_b in range(0, layer_index_a):

                    for hidden_neuron_b in self.hidden_layers[layer_index_b]:

                        link = [hidden_neuron_b.coordinates, 0.0]
                        links[hidden_neuron_b.neuron_id] = link

                hidden_neuron_a.links = links
