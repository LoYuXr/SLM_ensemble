import json

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  