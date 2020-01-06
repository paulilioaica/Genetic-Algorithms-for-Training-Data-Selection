import random

import numpy as np

random.seed(99)

POPULATION_SIZE = 10
EPOCHS = 30
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2
min_loss = 999
max_accuracy = 0
epoch_accuracy = [0 for i in range(EPOCHS)]
BEST_CHROMOSOME = None
K = 35


def check_population(sizes, population):
    for i, size in enumerate(sizes):
        if size == 0:
            population[i] = [1 for i in range(POPULATION_SIZE)]
    return population


def train(neural_nets, X_train, Y_train, X_test, Y_test, population):
    # train EPOCHS for each network and test it on test data
    global max_accuracy
    global min_loss
    global BEST_CHROMOSOME
    losses = [[] for i in range(POPULATION_SIZE)]
    accuracy = [[] for i in range(POPULATION_SIZE)]
    for epoch in range(EPOCHS):

        for j, net in enumerate(neural_nets):
            x_local = np.copy(X_train[j])
            net.train(x_local, np.expand_dims(np.array(Y_train[j]), axis=1))

        for j, net in enumerate(neural_nets):
            x_local = np.copy(X_test)
            output = net.predict(x_local)
            loss = np.square(np.argmax(output, axis=1) - Y_test).mean()
            acc = len(np.where(np.argmax(output, axis=1) == Y_test)[0]) / len(Y_test)
            accuracy[j].append(acc)
            losses[j].append(loss)

    avg_accuracy = [sum(x) / len(x) for x in accuracy]
    losses = [sum(x) / len(x) for x in losses]
    if max_accuracy < max(avg_accuracy):
        max_idx = np.argmax(avg_accuracy)
        max_accuracy = max(avg_accuracy)
        min_loss = losses[avg_accuracy.index(max(avg_accuracy))]
        BEST_CHROMOSOME = population[max_idx]
        print(BEST_CHROMOSOME)
        print(max_accuracy)
        epoch_accuracy[epoch] = avg_accuracy
        return avg_accuracy,max_accuracy, min_loss, [x for x in BEST_CHROMOSOME]

    epoch_accuracy[epoch] = avg_accuracy
    return avg_accuracy, max_accuracy, min_loss, None


def selection(accuracies, bin_population):
    first_candidate, second_candidate = (-np.array(accuracies)).argsort()[:2]
    return bin_population[first_candidate], bin_population[second_candidate]


def cross_over(first_candidate, second_candidate):
    random_pointk1 = np.random.randint(0, K)
    random_pointk2 = np.random.randint(K, 2 * K)
    random_pointk3 = np.random.randint(2 * K, 3 * K)
    for i in range(0, K):
        if i > random_pointk1:
            first_candidate[i], second_candidate[i] = second_candidate[i], first_candidate[i]
    for i in range(K, 2 * K):
        if i > random_pointk2:
            first_candidate[i], second_candidate[i] = second_candidate[i], first_candidate[i]
    for i in range(2 * K, 3 * K):
        if i > random_pointk3:
            first_candidate[i], second_candidate[i] = second_candidate[i], first_candidate[i]
    return first_candidate


def mutation(offspring, p=0.01):
    for i in range(len(offspring)):
        if np.random.rand() > 0.9:
            offspring[i] = int(not offspring[i])

    return offspring


def get_ds_from_population(X_train, X_bin):
    X_gen = []
    Y_gen = []
    for i in range(POPULATION_SIZE):
        c0, c1, c2 = X_bin[i][0:K], X_bin[i][K:2 * K], X_bin[i][2 * K:]
        x1 = [X_train[0][i] for i in range(len(c0)) if c0[i] == 1]
        x2 = [X_train[1][i] for i in range(len(c0)) if c1[i] == 1]
        x3 = [X_train[2][i] for i in range(len(c0)) if c2[i] == 1]
        X_gen.append(x1 + x2 + x3)
        y1 = [0 for i in range(len(x1))]
        y2 = [0 for i in range(len(x2))]
        y3 = [0 for i in range(len(x3))]
        Y_gen.append(y1 + y2 + y3)
    return X_gen, Y_gen


def generate_population(seed=0):
    random.seed(99)
    X_bin = []
    for i in range(POPULATION_SIZE):
        c0 = [random.choice([1, 0]) for j in range(K)]
        c1 = [random.choice([1, 0]) for j in range(K)]
        c2 = [random.choice([1, 0]) for j in range(K)]
        X_bin.append(c0 + c1 + c2)
    return X_bin
