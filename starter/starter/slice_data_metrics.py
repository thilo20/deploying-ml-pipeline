import pickle

import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference


def calculate_categorical_metrics(
    data, model, categorical_features, label, encoder, lb
):
    """
    Compute model performance on data slices for categorical features.

    For each categorical feature, compute metrics for each unique value.
    """
    results = []

    slice_min_size = data.shape[0] * 0.01  # Minimum slice size of 1% of data
    print(f"Calculating metrics for slices with minimum size of 1%: {slice_min_size}")

    # Loop through each categorical feature
    for feature in categorical_features:
        unique_values = data[feature].unique()

        # For each unique value of the feature
        for value in unique_values:
            slice_data = data[data[feature] == value]

            # Skip small slices
            if len(slice_data) < slice_min_size:
                continue

            # Process the slice
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Get predictions and metrics
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            results.append(
                {
                    "feature": feature,
                    "value": value,
                    "n_samples": len(slice_data),
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

    return results


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("../data/census.csv")

    # Load model
    with open("../model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("../model/lb.pkl", "rb") as f:
        lb = pickle.load(f)

    # Define categorical features
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

    # Compute slice metrics
    results = calculate_categorical_metrics(
        data, model, cat_features, "salary", encoder, lb
    )

    # Write to file
    df_results = pd.DataFrame(results)
    df_results.to_csv("slice_results.csv", index=False)

    print("Results written to slice_results.csv")
