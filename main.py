from data import get_data
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    print(f"{X_train.shape}----{X_val.shape}----{X_test.shape}")
