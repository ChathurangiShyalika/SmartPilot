import json
import time
from opcua import Client
import numpy as np
from flask import Flask, jsonify
import pickle
import Model
import threading
import os
import pandas as pd
import csv
import RCAFiles
from dotenv import load_dotenv
from RCAFiles.python_files.classes.neo4j_connection import Neo4jConnection
from RCAFiles.python_files.classes.ontology import Ontology
from RCAFiles.python_files.classes.reasoner import AnomalyReasoner
from flask_cors import CORS


app = Flask(__name__)
#CORS(app)  # Enable CORS for all routes


stop_event = threading.Event()

R02 = []
R03 = []
RCAData = []
CycleState = 0
load_dotenv()
URI = 'bolt://localhost:7687'
USER = 'neo4j'
PASSWORD = 'twine-grooves-accelerations'
# AUTH = (os.getenv("NEO4J_USER_NAME"), os.getenv("NEO4J_PASSWD"))

@app.route('/get_latest', methods=['GET'])
def get_latest():
    global R02, R03, CycleState
    try:
        output = jsonify({'latest_R02': R02[-1], 'latest_R03': R03[-1], 'CycleState': CycleState})
        return output
    except IndexError:
        print("No values yet")
        return jsonify({'latest_R02': None, 'latest_R03': None, 'CycleState': "No Values"})


@app.route('/predict', methods=['GET'])
def predict():
    global R02, R03, stop_event
    stop_event.set()  # Stop the data collection thread
    shape = (1, 2, 1250)
    TSC_Input = np.zeros(shape)
    # Interpolate and prepare data
    resampled_data = np.interp(np.linspace(0, len(R02), 1250), np.arange(len(R02)), R02)
    TSC_Input[0, 1, :] = resampled_data
    resampled_data_R03 = np.interp(np.linspace(0, len(R03), 1250), np.arange(len(R03)), R03)
    TSC_Input[0, 0, :] = resampled_data_R03
    with open('TSC_Input.pkl', 'wb') as w:
        pickle.dump(TSC_Input, w)
    prediction = Model.predictAnomaly(TSC_Input)
    if prediction is not None:
        return jsonify({'Prediction': prediction})
    else:
        return jsonify({'Prediction': "Analyzing"})


def collect_data():
    global R02, R03, CycleState, RCAData, stop_event
    server_endpoint = "opc.tcp://192.168.0.2:4840"
    client = Client(server_endpoint)
    client.connect()
    node_inputs = client.get_node("ns=3;s=Inputs")
    node_outputs = client.get_node("ns=3;s=Outputs")
    while not stop_event.is_set():
        CycleState = node_outputs.get_child("Q_Cell_CycleState").get_value()
        R03_Pot = node_inputs.get_child("I_R03_Gripper_Pot").get_value()
        R02_Pot = node_inputs.get_child("I_R02_Gripper_Pot").get_value()
        R02_Load = node_inputs.get_child("I_R02_Gripper_Load").get_value()
        R03_Load = node_inputs.get_child("I_R03_Gripper_Load").get_value()
        R01_Pot = node_inputs.get_child("I_R01_Gripper_Pot").get_value()
        R01_Load = node_inputs.get_child("I_R01_Gripper_Load").get_value()
        if CycleState in range(0, 9):
            R02.append(R02_Pot)
            R03.append(R03_Pot)
            sensor_variables = {
                "I_R01_Gripper_Load": R01_Load,
                "I_R02_Gripper_Load": R02_Load,
                "I_R03_Gripper_Load": R03_Load,
                "I_R01_Gripper_Pot": R01_Pot,
                "I_R02_Gripper_Pot": R02_Pot,
                "I_R03_Gripper_Pot": R03_Pot,
                # Add more sensor variables as needed
            }
            structured_data = {
                "cycle_state": CycleState,
                "sensor_variables": sensor_variables
            }
            RCAData.append(structured_data)
        time.sleep(1)  # Add a small sleep to reduce CPU usage

def get_formatted_data(filepath):
    """
    Return the data in the following format
    [ {'cycle_state': <value>,
    'sensor_variables':{'I_R01_Gripper_Load':<value>,
                        'I_R02_Gripper_Load':<value>,
                        'I_R04_Gripper_Load':<value>,...}
    }, {'cycle_state':<value>, "",..}..]
    # number of dict = number of rows in csv/df
    """
    data_list = []
    df = pd.read_csv(filepath)
    df = df.drop(labels=['Description', '_time'], axis=1) #0-rows, 1-columns
    headers = df.columns.tolist()
    for i in range(0,len(df)):
        data_dict = {}
        sensor_dict = {}
        for header in headers:
            if header == 'CycleState':
                data_dict['cycle_state'] = df[header][i]
            else:
                sensor_dict[str(header)] = df[header][i]
        data_dict['sensor_variables'] = sensor_dict
        data_list.append(data_dict)

    return data_list


def save_specified_values_if_changed(df):
    saved_values = []
    seen_entries = set()

    for idx, row in df.iterrows():
        entry = (
            row['anomalous_sensor_variables'],
            row['robot_names'],
            row['cycle_function'],
            row['cycle_state'],
            row['sensor_names']
        )

        if entry not in seen_entries:
            saved_values.append({
                "anomalous_sensor_variables": row['anomalous_sensor_variables'],
                "robot_names": row['robot_names'],
                "cycle_function": row['cycle_function'],
                "cycle_state": row['cycle_state'],
                "sensor": row['sensor_names']
            })
            seen_entries.add(entry)

    return saved_values

@app.route('/RootCause', methods=['GET'])
def RootCause():
    neo4j_obj = Neo4jConnection(uri=URI,
                                user=USER,
                                pwd=PASSWORD)
    # Specify the filepaths
    ontology_filepath = "../mfg-data/process_ontology.txt"  # filepath that consists of ontology creation query
    min_max_filepath = '../mfg-data/cycle_state_values.csv'  # filepath that consists of min and max values of sensors as per cycle state
    cycle_function_path = '../mfg-data/cycle_state_function.csv'
    # create an object for ontology class
    ont = Ontology()

    # Inject ontology to Neo4j when empty
    nodes = neo4j_obj.query("MATCH (n) RETURN n")
    if len(nodes) == 0:
        # TODO - create constraint
        res = ont.create(neo4j_obj, ontology_filepath)
        print("Result of Ontology Creation:", res)

        # get the data in required format to update ontology
        min_max_data = RCAFiles.python_files.main.get_min_max_data(filepath=min_max_filepath)
        # call the update function for min max
        res = ont.update_min_max(neo4j_obj, min_max_data)
        print("Min Max value Update:", res)

        # get the cycle functions and load it into the ontology
        cycle_function_data = RCAFiles.python_files.main.get_cycle_function_data(filepath=cycle_function_path)
        # call the required function
        res = ont.add_cycle_functions(neo4j_obj, cycle_function_data)
        print("Adding Cycle Functions:", res)
    else:
        pass

    ############## ONTOLOGY USAGE ##########
    """
    The following code will generate reasoning. Now that the ontology is created, you can use 
    it for explanations
    """

    # get the data for anomalous cycle
    # NOTE: Input file must contain only the cycle state and sensor values
    anomalous_data_filepath = 'RCAFiles/mfg-data/anomaly_data/fadi2.csv'
    anomalous_data = get_formatted_data(anomalous_data_filepath)
    # anomalous_data = RCAData

    # get the explanation for anomaly
    # Instantiate Reasoner class
    reasoner = AnomalyReasoner()
    exp_dict = reasoner.get_explanation(neo4j_obj, anomalous_data)
    # store the values in a csv file
    df = pd.DataFrame.from_dict(exp_dict)
    anomalies = save_specified_values_if_changed(df)
    print(anomalies)
    return anomalies


if __name__ == '__main__':
    stop_event.clear()  # Ensure the event is cleared before starting the thread
    #
    # data_thread = threading.Thread(target=collect_data)
    # data_thread.daemon = True
    # data_thread.start()

    # Run the Flask application
    app.run(port=5002, debug=True)
