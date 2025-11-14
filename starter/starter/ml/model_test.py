import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def sample_data():
    """
    Fixture to create sample training and test data.
    """
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def trained_model(sample_data):
    """
    Fixture to create a trained model.
    """
    X_train, y_train, _, _ = sample_data
    model = train_model(X_train, y_train)
    return model


# Tests for train_model function
def test_train_model_returns_random_forest(sample_data):
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X_train, y_train, _, _ = sample_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_train_model_is_fitted(sample_data):
    """
    Test that the returned model is fitted and can make predictions.
    """
    X_train, y_train, X_test, _ = sample_data
    model = train_model(X_train, y_train)
    # If model is fitted, it should be able to predict without error
    predictions = model.predict(X_test)
    assert predictions is not None
    assert len(predictions) == len(X_test)


def test_train_model_with_different_data_shapes():
    """
    Test that train_model works with different data shapes.
    """
    X_train = np.random.rand(50, 3)
    y_train = np.random.randint(0, 2, 50)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_train_model_reproducibility():
    """
    Test that train_model produces consistent results with the same data
    due to random_state.
    """
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)

    model1 = train_model(X_train, y_train)
    model2 = train_model(X_train, y_train)

    X_test = np.random.rand(20, 5)
    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)

    assert np.array_equal(preds1, preds2)


# Tests for compute_model_metrics function
def test_compute_model_metrics_perfect_prediction():
    """
    Test compute_model_metrics with perfect predictions.
    """
    y_true = np.array([1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_compute_model_metrics_imperfect_prediction():
    """
    Test compute_model_metrics with imperfect predictions.
    """
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check that metrics are between 0 and 1
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0

    # With this data: TP=3, FP=1, FN=1
    # Precision = 3/4 = 0.75
    # Recall = 3/4 = 0.75
    # F1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    assert precision == 0.75
    assert recall == 0.75
    assert fbeta == 0.75


def test_compute_model_metrics_returns_three_values():
    """
    Test that compute_model_metrics returns exactly three values.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    result = compute_model_metrics(y_true, y_pred)

    assert len(result) == 3
    assert all(isinstance(val, (float, np.floating)) for val in result)


# Tests for inference function
def test_inference_returns_predictions(trained_model, sample_data):
    """
    Test that inference returns predictions.
    """
    _, _, X_test, _ = sample_data
    predictions = inference(trained_model, X_test)

    assert predictions is not None
    assert len(predictions) == len(X_test)


def test_inference_prediction_shape(trained_model, sample_data):
    """
    Test that inference returns predictions with correct shape.
    """
    _, _, X_test, _ = sample_data
    predictions = inference(trained_model, X_test)

    assert predictions.shape == (len(X_test),)


def test_inference_binary_predictions(trained_model, sample_data):
    """
    Test that inference returns binary predictions (0 or 1).
    """
    _, _, X_test, _ = sample_data
    predictions = inference(trained_model, X_test)

    assert all(pred in [0, 1] for pred in predictions)


def test_inference_with_single_sample(trained_model):
    """
    Test that inference works with a single sample.
    """
    X_single = np.random.rand(1, 5)
    predictions = inference(trained_model, X_single)

    assert len(predictions) == 1
    assert predictions[0] in [0, 1]


def test_inference_with_multiple_samples(trained_model):
    """
    Test that inference works with multiple samples.
    """
    X_multiple = np.random.rand(50, 5)
    predictions = inference(trained_model, X_multiple)

    assert len(predictions) == 50
    assert all(pred in [0, 1] for pred in predictions)


def test_inference_deterministic(trained_model):
    """
    Test that inference produces consistent results for the same input.
    """
    X_test = np.random.rand(10, 5)

    predictions1 = inference(trained_model, X_test)
    predictions2 = inference(trained_model, X_test)

    assert np.array_equal(predictions1, predictions2)
