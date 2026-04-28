import urllib.request, json

# Test health
r = urllib.request.urlopen("http://127.0.0.1:5001/health")
print("Health:", json.dumps(json.loads(r.read()), indent=2))

# Test prediction
data = json.dumps({
    "Client_Income": 25000,
    "Credit_Amount": 180000,
    "Loan_Annuity": 9000,
    "Age_Days": -14000,
    "Employed_Days": -2000,
    "Car_Owned": 1,
    "Bike_Owned": 0,
    "Active_Loan": 1,
    "Child_Count": 2,
    "Client_Gender": "Male",
    "Loan_Contract_Type": "CL",
    "Client_Family_Members": 3
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:5001/predict",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
r = urllib.request.urlopen(req)
print("\nPrediction:", json.dumps(json.loads(r.read()), indent=2))