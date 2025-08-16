import requests

url = "http://127.0.0.1:5000/predict"   # Flask API endpoint

# Example test input (adjust values if needed)
data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 22.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 200.0
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
