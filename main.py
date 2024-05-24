from data import get_data
from .settings import *
import keras


def data_augmentation(X_train):
    data_augmentation = keras.Sequential(
        [
            keras.layers.Normalization(),
            keras.layers.Resizing(image_size, image_size),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.02),
            keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(X_train)

    return data_augmentation


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    print(f"{X_train.shape}----{X_val.shape}----{X_test.shape}")
