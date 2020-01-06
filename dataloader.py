import numpy as np


def load_dataset(name='wine.data'):
    with open(name, 'r') as f:
        dataset = [line.rstrip() for line in f.readlines()]

    return dataset


def normalize(X):
    X = np.array(X)
    for i in range(13):
        total_sum = np.sum(X[:, i])
        avg = total_sum / len(X[:, i])
        var = np.var(X[:, i])
        X[:, i] -= avg
        X[:, i] /= var
    return X


def parse_dataset():
    dataset = load_dataset()
    test_dataset = np.random.choice(dataset, int(0.15 * len(dataset)), replace=False)
    train_dataset = list(set(dataset) - set(test_dataset))
    X_train = normalize([[float(x) for x in data.split(",")][1:] for data in train_dataset])
    X_test = normalize([[float(x) for x in data.split(",")][1:] for data in test_dataset])

    Y_train = np.array([int(x.split(',')[0]) - 1 for x in train_dataset])
    Y_test = [int(x.split(',')[0]) - 1 for x in test_dataset]

    X_train_c0 = X_train[Y_train == 0]
    X_train_c1 = X_train[Y_train == 1]
    X_train_c2 = X_train[Y_train == 2]
    X_train = [X_train_c0, X_train_c1, X_train_c2]

    return X_train, np.expand_dims(np.array(Y_train), axis=1), normalize(X_test), np.array(Y_test)
