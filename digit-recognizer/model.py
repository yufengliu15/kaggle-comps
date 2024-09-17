# submission is 1 index based
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def read_data():
    X = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y_train = X["label"]
    X = X.drop(columns=["label"])

    #print("X: \n", X.head())
    #print("y: \n", y_train.head())

    #print(f"Shape: {X.shape}")

    return X, y_train, test

def train(X, y_train):
    model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 64), max_iter=30 ,random_state=1, batch_size=32, verbose=True)
    model.fit(X, y_train)
    loss_values = model.loss_curve_

    # Print or plot the loss values
    print(loss_values)

    return model

if __name__ == '__main__':
    X_train, y_train, test = read_data()

    model = train(X_train, y_train)

    y_pred = model.predict(test)

    output = pd.DataFrame(data=y_pred)

    output.index += 1
    print(output.head())
    output.to_csv("result.csv")


    #accuracy = accuracy_score(y_train, y_pred)
    #print(f"Accuracy: {accuracy:.2f}")