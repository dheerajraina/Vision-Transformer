from data import get_data
from model import VisionTransformer


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    vit = VisionTransformer(X_train, y_train, X_test, y_test)
    history = vit.run_experiment(vit)
    vit.plot_history(history, "loss")
