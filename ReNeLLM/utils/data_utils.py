import csv, json, re

def data_reader(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        goals = []
        for row in reader:
            goal = row[0]
            goals.append(goal)
    return goals

def jailbroken_data_reader(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def remove_number_prefix(sentence):
    return re.sub(r'^\d+\.\s*', '', sentence)