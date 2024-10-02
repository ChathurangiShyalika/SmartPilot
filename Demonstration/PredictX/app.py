from flask import Flask, render_template, request, jsonify
import numpy as np
import requests


app = Flask(__name__, static_url_path='/static')

CycleState = 0
r02 = 0
r03 = 0
R02 = []
R03 = []
predicting = False
@app.route('/TSC.html')
def TSC():
    data = np.load("PotData.npy")
    data_list = data.tolist()
    latest_value = data_list[-1]
    return render_template('TSC.html', latest_value=latest_value)

@app.route('/NS_HyMAP.html')
def NS_HyMAP():
    return render_template('/NS_HyMAP.html')

@app.route('/RCA.html')
def RCA():
    return render_template('RCA.html')

@app.route('/RootCause')
def RootCause():
    response = requests.get("http://localhost:5002/RootCause")
    return response.json()
    # return AnomalousSensors, AnomalousRobots, AnomalousCycleState

@app.route('/get_latest')
# Assume `get_latest_data()` is a function that returns the latest values of R02 and R03
def get_latest_data():
    global r02, r03
    if not predicting:
        response = requests.get("http://localhost:5002/get_latest")
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the latest_R02 value from the JSON response
            CycleState = response.json()['CycleState']
            print(CycleState)
            latest_R02 = response.json()['latest_R02']
            latest_R03 = response.json()['latest_R03']
            return latest_R02, latest_R03, CycleState
        else:
            # Print an error message if the request was unsuccessful
            print("Error:", response.status_code)
            return [0, 0, "N/A"]  # Placeholder for demonstration purposes
    else:
        return [r02, r03, "9"]

@app.route('/api/data')
def api_data():
    global r02, r03, CycleState
    r02, r03, CycleState = get_latest_data()
    return jsonify({'r02': r02, 'r03': r03, 'CycleState': CycleState})

@app.route('/predict')
def predict():
    global predicting
    predicting = True
    response = requests.get("http://localhost:5002/predict")
    if response.status_code == 200:
        print("In predict API")
        prediction = response.json()['Prediction']
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'prediction': "Analyzing"})


if __name__ == '__main__':
    app.run(debug=True)
