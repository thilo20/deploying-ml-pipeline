import requests


def post_call(url):
    """Test POST predicts >50K for high earner profile"""
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }

    response = requests.post(url, json=data)
    print(f"statuscode: {response.status_code}")
    print(response.json())


if __name__ == "__main__":
    print("Calling the live API on Heroku..")
    url = "https://thilo-three-a8760c136c8a.herokuapp.com/predict"
    post_call(url)
