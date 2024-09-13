'''
What have I learned by doing this comp?
- Data preprocessing is very important
    - Check for nulls
    - Normalize
    - Get rid of strings
        - Not sure how to represent numerically tho...
- If you normalize training data, then normalize test data too. 
- Threshold evaluation
    - Accuracy vs Precision vs Recall vs F1
        - In this case, i will choose the intersection between precision and recall
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

scaler = StandardScaler()

# model returns decimal values...
# how would i know what it thinks is a yes or no?
def evaluate_threshold(y_train, y_pred):
    # Assuming you have these from your model:
    # y_true: true labels (0 or 1)
    # y_pred_proba: predicted probabilities

    def evaluate_threshold(y_true, y_pred_proba, threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    # Evaluate range of thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = [evaluate_threshold(y_train, y_pred, t) for t in thresholds]

    # Plot results
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.plot(thresholds, [r[metric] for r in results], label=metric)

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Scores at Different Thresholds')
    plt.legend()
    plt.show()

    # Find threshold with highest F1 score (for example)
    best_f1_threshold = max(results, key=lambda x: x['f1'])
    print(f"Best threshold for F1 score: {best_f1_threshold['threshold']}")
    print(f"F1 score at this threshold: {best_f1_threshold['f1']}")

def retrieve_data(filename, test):
    data = pd.read_csv(filename)

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
    else:
        return scaler.transform(X), data["PassengerId"]

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
X_test, passenger_id_df = retrieve_data('test.csv', True)

model = init_model()
train_model(model)

predictions = model.predict(X_test)
y_train_pred = model.predict(X_train)
print(predictions.shape)

# 0.42
threshold = 0.42
#evaluate_threshold(y_train, y_train_pred)

predictions = pd.DataFrame(predictions, columns = ["Survived"])

result = pd.concat([passenger_id_df, predictions], axis=1)

result["Survived"][result["Survived"] > threshold] = 1
result["Survived"][result["Survived"] <= threshold] = 0
result.reset_index(drop=True, inplace=True)

result.to_csv("result.csv")
print(f"Predictions: {result[:10]}")
print(f"shape: {result.shape}")