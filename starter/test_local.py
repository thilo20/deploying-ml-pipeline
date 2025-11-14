from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_api_locally_ingest_valid_data():
    data = {"feature_1": 1.0, "feature_2": "valid string"}
    r = client.post("/ingest", json=data)
    assert r.status_code == 200
    assert r.json() == {"received_data": data}


def test_api_locally_ingest_invalid_data():
    data = {"feature_1": -1.0, "feature_2": "valid string"}
    r = client.post("/ingest", json=data)
    assert r.status_code == 400
    assert "error" in r.json()["detail"]


def test_api_locally_ingest_invalid_data2():
    data = {"feature_1": 1.0, "feature_2": "a" * 281}  # feature_2 too long
    r = client.post("/ingest", json=data)
    assert r.status_code == 400
    assert "error" in r.json()["detail"]
