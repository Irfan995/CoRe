import torch 
from datetime import datetime

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_datetime_str():
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string

def calculate_layer_input(dimension):
    filter = (dimension * 2) / 4
    input_layer = dimension
    hidden_layer_one = (dimension * 2) / 3
    hidden_layer_two = dimension / 3
    
    return filter, input_layer, hidden_layer_one, hidden_layer_two
