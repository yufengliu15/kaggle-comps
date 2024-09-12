import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

def retrieve_data(filename, test):
    data = pd.read_csv(filename)

    scaler = StandardScaler()

    if not test:
        X = data.drop(['Survived', 'Name', 'Cabin', 'Ticket'], axis=1)
        y = data['Survived']
    else:
        X = data.drop(['Name', 'Cabin', 'Ticket'], axis=1)
        y = None

    # one-hot encoding??
    for i in range(len(X["Sex"])): 
        if X.at[i, "Sex"] == "male":
            X.at[i, "Sex"] = 1
        else:
            X.at[i, "Sex"] = 2
    
    X["Sex"] = X["Sex"].astype(int)


    for i in range(len(X["Embarked"])): 
        if X.at[i, "Embarked"] == "C":
            X.at[i, "Embarked"] = 1
        elif X.at[i, "Embarked"] == "Q":
            X.at[i, "Embarked"] = 2
        elif X.at[i, "Embarked"] == "S":
            X.at[i, "Embarked"] = 3
        else: 
            X.at[i, "Embarked"] = 1

    X["Embarked"] = X["Embarked"].astype(int)

    # if values are null
    for col in X:
        X[col].fillna(value=X[col].mean(), inplace=True)

    if not test:
        return scaler.fit_transform(X), y
    
    return X

def init_model():
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(8,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model):
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)


X_train, y_train = retrieve_data('train.csv', False)
X_test = retrieve_data('test.csv', True)

model = init_model()
train_model(model)

predictions = model.predict(X_test)
print(predictions.shape)

predictions = pd.DataFrame(predictions, columns = ["Survived"])
passenger_id_df = X_test["PassengerId"]

result = pd.concat([passenger_id_df, predictions], axis=1)
result.reset_index(drop=True, inplace=True)

for key in result: result[key].to_csv("result.csv")
print(f"Predictions: {result[:10]}")
print(f"shape: {result.shape}")