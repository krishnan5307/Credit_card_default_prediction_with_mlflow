from flask import Flask, request
import sys
import numpy as np, pandas as pd
import pip
from insurance.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from insurance.logger import logging
from insurance.exception import InsuranceException
import os, sys
import json
from insurance.config.configuration import Configuartion
from insurance.constant import CONFIG_DIR, get_current_time_stamp
from insurance.pipeline.pipeline import Pipeline
from insurance.entity.premium_predictor import InsurancePredictor, InsuranceData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "insurance"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from insurance.logger import get_log_dataframe

INSURANCE_DATA_KEY = "insurance_data"
EXPENSES_VALUE_KEY = "expenses"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)




@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {                               ## declared outside of request.method for to be used both after and before submiting form and view html predict.html
        INSURANCE_DATA_KEY: None,
        EXPENSES_VALUE_KEY: None
    }

    if request.method == 'POST':      ## accessed when html form sends request while submiting form 
                                      ## html form has inbuilt request sending methods Post, Get etc and action(eg - /predict)
                                      ## Corresponds to the HTTP POST or GET method; form data are included in the body of the form and sent to the server.
        age = int(request.form['age'])
        children = int(request.form['children'])
        bmi = float(request.form['bmi'])
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']
        insurance_data = InsuranceData(age = age,
                                     children = children,
                                     bmi = bmi,
                                     sex = sex,
                                     smoker = smoker,
                                     region = region,
                                       ) 
        
        insurance_df = insurance_data.get_insurance_input_data_frame() ## calling function inside hosuing class
        premium_predictor = InsurancePredictor(model_dir=MODEL_DIR)      ## creating an object with intialization as model_dir
        expenses = premium_predictor.predict(X=insurance_df)   ## using the above obj to do preiction
        context = {
            INSURANCE_DATA_KEY: insurance_data.get_insurance_data_as_dict(), ## FOR PASSING TO HTML PAGE VIA CONTEXT
            EXPENSES_VALUE_KEY: expenses,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)     ## to display html when  '/predict'


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)             ## downloaidng file if its a file like model.pkl

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}   ## else go in inside folder

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)





if __name__ == "__main__":
    app.run(debug=True)
