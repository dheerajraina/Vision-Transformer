import keras
from sklearn.model_selection import train_test_split


def get_data():
    (X, y), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=134)

    return X_train, X_val, X_test, y_train, y_val, y_test
