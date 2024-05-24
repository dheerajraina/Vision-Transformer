import keras


def get_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

    return X_train, X_test, y_train, y_test
