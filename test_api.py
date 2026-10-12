import requests

url = "http://127.0.0.1:5000/predict"

form_data = {
    'country': 'Algeria',
    'gender': '0',
    'age': '41.8',
    'annual_salary': '62812.0',
    'credit_card_debt': '11609.0',
    'net_worth': '238961.0'
}

response = requests.post(url, data=form_data)
print(response.json())