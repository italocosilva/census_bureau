import requests
import json

if __name__ == '__main__':
    url = 'https://italocosilva-census-bureau.herokuapp.com/inference/'

    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    r = requests.post(url, data=json.dumps(data))
    print(f'Status code: {r.status_code}')
    print(f'Response: {r.text}')
