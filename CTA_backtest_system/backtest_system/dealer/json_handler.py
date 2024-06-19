import csv
import json
import sys
from pathlib import Path
import os

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
    # create a dictionary
    data = {}
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
         
        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
            # Assuming a column named 'No' to
            # be the primary key
            key = rows['uid']
            data[key] = rows
 
    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

csvFilePath = f'{user_home_path}' 
jsonFilePath =  f'{user_home_path}'+'/backtest_system/backtest_engine/backtest_initiate/result.json'
 

savedir = f'{user_home_path}' + '/backtest_system/backtest_engine/backtest_initiate'

def save_json():
    with open ( f'{user_home_path}'+ '/backtest_system/backtest_engine/backtest_initiate/result.json') as fp:
        datadict = json.load(fp)
        for item in datadict:
            print(datadict[item])
            #/Users/johnsonhsiao/990.json
            with open(savedir + '/' + str(item) + '.json', 'w') as jsonf:
                json.dump(datadict[item], jsonf)     

if __name__ == '__main__':
    make_json(csvFilePath, jsonFilePath)
    save_json()
