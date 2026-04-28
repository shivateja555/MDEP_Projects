import urllib.request, json, urllib.error

# Send all columns the model expects - missing ones set to null
data = json.dumps({
    "Client_Income": 25000,
    "Credit_Amount": 180000,
    "Loan_Annuity": 9000,
    "Age_Days": -14000,
    "Employed_Days": -2000,
    "Registration_Days": -5000,
    "ID_Days": -1000,
    "Car_Owned": 1,
    "Bike_Owned": 0,
    "Active_Loan": 1,
    "House_Own": 0,
    "Child_Count": 2,
    "Client_Family_Members": 3,
    "Client_Gender": "Male",
    "Loan_Contract_Type": "CL",
    "Client_Income_Type": "Service",
    "Client_Education": "Secondary",
    "Client_Marital_Status": "M",
    "Client_Housing_Type": "Home",
    "Accompany_Client": "Alone",
    "Client_Occupation": "Laborers",
    "Type_Organization": "Business Entity Type 3",
    "Mobile_Tag": 1,
    "Homephone_Tag": 0,
    "Workphone_Working": 1,
    "Client_Permanent_Match_Tag": 1,
    "Client_Contact_Work_Tag": 1,
    "Phone_Change": 1000,
    "Score_Source_1": None,
    "Score_Source_2": 0.5,
    "Score_Source_3": None,
    "Social_Circle_Default": None,
    "Credit_Bureau": 0,
    "Application_Process_Hour": 10,
    "Application_Process_Day": 2,
    "Population_Region_Relative": 0.02,
    "Own_House_Age": None,
    "Cleint_City_Rating": 2,
    "House_Own": 0
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:5001/predict",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    r = urllib.request.urlopen(req)
    print("Success:")
    print(json.dumps(json.loads(r.read()), indent=2))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Server says:", e.read().decode())
except Exception as e:
    print("Error:", e)