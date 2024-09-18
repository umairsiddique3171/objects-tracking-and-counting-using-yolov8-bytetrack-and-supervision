import yaml 
import json
from box import Box
import matplotlib.pyplot as plt


def load_yaml(file_path):
    with open(file_path,"r") as file:
        return Box(yaml.safe_load(file))
    
def load_json(file_path):
    with open(file_path,"r") as file:
        data = json.load(file)
    data_ = {int(k): v for k, v in data.items()}
    return data_


