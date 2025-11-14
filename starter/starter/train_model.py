# Script to train machine learning model.

import pickle

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
from sklearn.model_selection import train_test_split

# Add code to load in the cleaned data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=44)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
# Train and save a model.
print("Training model...")
model = train_model(X_train, y_train)

# Evaluate on test set
print("Evaluating model...")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Model Performance on test data set:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F-beta: {fbeta:.4f}")

# Save the model and encoders
print("Saving model...")
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

print("Model training finished. Files stored in ../model/")
