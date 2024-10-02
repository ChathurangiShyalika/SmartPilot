import pandas as pd


data_list = [{'cycle_state': 1, 'I_R01_Gripper_Pot': {'min': '4500 ', 'max': ' 12000'}, 'I_R01_Gripper_Load': {'min': '0 ', 'max': ' 1000'}, 'I_R02_Gripper_Pot': {'min': '11500 ', 'max': ' 12000'}, 'I_R02_Gripper_Load': {'min': '1200', 'max': '1400'}, 'I_R03_Gripper_Pot': {'min': '2200', 'max': '2500'}, 'I_R03_Gripper_Load': {'min': '1300', 'max': '1450'}, 'I_R04_Gripper_Pot': {'min': '12000', 'max': '13000'}, 'I_R04_Gripper_Load': {'min': 'NA', 'max': 'NA'}}]

for item in data_list:
    print(list(item.keys()))