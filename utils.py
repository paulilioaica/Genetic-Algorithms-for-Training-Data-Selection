from net import NeuralNetwork

POPULATION_SIZE = 5

def create_nn_array(sizes):
    neural_nets = [NeuralNetwork(input_size=13, hidden_size=13, output_size=3) for i in range(len(sizes))]
    return neural_nets


def get_sizes(population):
    return [x.count(1) for x in population]


def get_probab(losses):
    losses = [x / sum(losses) for x in losses]
    return losses

