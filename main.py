import random
from dataloader import parse_dataset
from genetic import train, selection, cross_over, mutation, generate_population, get_ds_from_population
from utils import get_sizes, create_nn_array

random.seed(99)
POPULATION_SIZE = 10
EPOCHS = 5
CHROMOSOME_SIZE = 13
GENERATIONS = 200
NUM_OFFSPRING = 2
X_train, Y_train, X_test, Y_test = parse_dataset()
accuracies = []
chromosomes = []
losses = []

features = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']


def run_genetic_algorithm():
    generation = 0
    bin_population = generate_population(X_train)
    for i in range(GENERATIONS):
        X_train_local, Y_train_local = get_ds_from_population(X_train, bin_population)
        sizes = get_sizes(bin_population)  # get the sizes of the new populations for the networks inputs
        nets = create_nn_array(sizes)  # we create a list of neural networks with predefined size
        accuracy,max_acc,min_loss, BEST_CHROMOSOME = train(nets, X_train_local, Y_train_local, X_test, Y_test, bin_population) # we train EPOCHS number for each network
        first_candidate, second_candidate = selection(accuracy, bin_population)
        new_candidate = cross_over(first_candidate, second_candidate)
        new_candidate = mutation(new_candidate, p=0.01)
        bin_population = [new_candidate]
        bin_population += [mutation(new_candidate, p=0.2) for i in range(POPULATION_SIZE-1)]
        generation += 1
        accuracies.append(max_acc)
        losses.append(min_loss)
        print("Training on generation {}".format(generation))

run_genetic_algorithm()
