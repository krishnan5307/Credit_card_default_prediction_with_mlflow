from flask import Flask, request
import sys
import numpy as np, pandas as pd
import pip
from credit_card_defaulters.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from credit_card_defaulters.logger import logging
from credit_card_defaulters.exception import CreditException
import os, sys
import json
from credit_card_defaulters.config.configuration import Configuartion
from credit_card_defaulters.constant import CONFIG_DIR, get_current_time_stamp
from credit_card_defaulters.pipeline.pipeline import Pipeline
from credit_card_defaulters.entity.premium_predictor import  CreditData, CreditPredictor
from flask import send_file, abort, render_template
from urllib.parse import urlparse
from dotenv import load_dotenv
import joblib
import mlflow

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "credit_card_defaulters"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from credit_card_defaulters.logger import get_log_dataframe

CREDIT_DATA_KEY = "credit_data"
CREDIT_VALUE_KEY = "defaulter_status"

MLFLOW_URI = "https://dagshub.com/krishnan5307/Credit_card_default_prediction_with_mlflow.mlflow"


## mlflow setup for main entry point python file ...here app.py , so it will run as long as application runs
mlflow.set_registry_uri(MLFLOW_URI)
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

# Load mlflow environment variables from configuration file
load_dotenv('mlflow.env')

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
        CREDIT_DATA_KEY: None,
        CREDIT_VALUE_KEY: None
    }

    if request.method == 'POST':      ## accessed when html form sends request while submiting form 
                                      ## html form has inbuilt request sending methods Post, Get etc and action(eg - /predict)
                                      ## Corresponds to the HTTP POST or GET method; form data are included in the body of the form and sent to the server.
       ## The html predict page will have form with fileds of the variables where we can type the text 
       ## and this text when got submited (button) will call this POST mehtod where we can access
       ## the data in the form we gave using below code given below

        LIMIT_BAL = float(request.form['LIMIT_BAL'])
        SEX = request.form['SEX']
        EDUCATION = request.form['EDUCATION']
        MARRIAGE = request.form['MARRIAGE']
        AGE = int(request.form['AGE'])
        PAY_0 = int(request.form['PAY_0'])
        PAY_2 = int(request.form['PAY_2'])
        PAY_3 = int(request.form['PAY_3'])
        PAY_4 = int(request.form['PAY_4'])
        PAY_5 = int(request.form['PAY_5'])
        PAY_6 = int(request.form['PAY_6'])
        BILL_AMT1 = float(request.form['BILL_AMT1'])
        BILL_AMT2 = float(request.form['BILL_AMT2'])
        BILL_AMT3 = float(request.form['BILL_AMT3'])
        BILL_AMT4 = float(request.form['BILL_AMT4'])
        BILL_AMT5 = float(request.form['BILL_AMT5'])
        BILL_AMT6 = float(request.form['BILL_AMT6'])
        PAY_AMT1 = float(request.form['PAY_AMT1'])
        PAY_AMT2 = float(request.form['PAY_AMT2'])
        PAY_AMT3 = float(request.form['PAY_AMT3'])
        PAY_AMT4 = float(request.form['PAY_AMT4'])
        PAY_AMT5 = float(request.form['PAY_AMT5'])
        PAY_AMT6 = float(request.form['PAY_AMT6'])
        ## now we need to replace integer keys for SEX, MARRIAGE a&nd EDUCATION and pass this data 
        ## to CreditData class to predict the output

        sex = {"male":1, "female":2}
        education = {"graduate school":1, "university":2, "high school":3, "others":4, "unknown":5}
        marriage = {"married":1, "single":2, "others":3}
        credit_data = CreditData(            ##CreditData class obj createion
                                LIMIT_BAL = LIMIT_BAL,
                                SEX = sex[SEX],
                                EDUCATION = education[EDUCATION],
                                MARRIAGE = marriage[MARRIAGE],
                                AGE = AGE,
                                PAY_0 = PAY_0,
                                PAY_2 = PAY_2,
                                PAY_3 = PAY_3,
                                PAY_4 = PAY_4,
                                PAY_5 = PAY_5,
                                PAY_6 = PAY_6,
                                BILL_AMT1 = BILL_AMT1,
                                BILL_AMT2 = BILL_AMT2,
                                BILL_AMT3 = BILL_AMT3,
                                BILL_AMT4 = BILL_AMT4,
                                BILL_AMT5 = BILL_AMT5,
                                BILL_AMT6 = BILL_AMT6,
                                PAY_AMT1 = PAY_AMT1,
                                PAY_AMT2 = PAY_AMT2,
                                PAY_AMT3 = PAY_AMT3,
                                PAY_AMT4 = PAY_AMT4,
                                PAY_AMT5 = PAY_AMT5,
                                PAY_AMT6 = PAY_AMT6
                                                ) 
        
        credit_df = credit_data.get_credit_input_data_frame() ## calling function inside CreditData class
        credit_predictor = CreditPredictor(model_dir=MODEL_DIR)      ## creating an object with intialization as model_dir
        defaulter_cla = credit_predictor.predict(X=credit_df)   ## calling fucntion in that using CreditPredictor class, the above obj to do preiction
        print(f"output of predicetd model: {defaulter_cla}")
        defaulter_class = defaulter_cla[0,1]*100 ## accessing 1st row and 2nd column in 2d numpy array
    # defaulter_stat= int(defaulter_status)
        ## converting defaulter status into readable form now 
    # defaulter = {1:"Default payment (Yes, the customer defaulted)", 0:"No default payment (No, the customer did not default)"}
        ## Now we need to pass the results predicted back to the HTML via context- Thats why predict.html has 2 sections
        ## one to take the input for predciton asn next section to show the predcited output
        ## initially context value will be None before submitting form and  context value will update to 
        ## below after predciting the value .
        context = {
            CREDIT_DATA_KEY: credit_data.get_credit_data_as_dict(), ## FOR PASSING TO HTML PAGE VIA CONTEXT
            CREDIT_VALUE_KEY: round(defaulter_class, 3) #int(defaulter_class*100) 
            ###  defaulter[defaulter_stat],
        }
        print(f"probability of user being defaulter: {round(defaulter_class, 3)}%")  
        ## 
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



## by declaring __main__: method we can execute the python code under the main method- by direct all eg: app.py
## so here in below file when app.py gets executed, 
## 1. load_dotenv('mlflow.env')
## 2. MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
   # setting up the mlflow regirsty with our tracking uri 
   # mlflow.set_registry_uri(MLFLOW_URI)
   # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
## 3. app.run(debug=True) gets executed, so the application with name app runs 

if __name__ == "__main__":
    # Load mlflow environment variables from configuration file into os env 
    load_dotenv('mlflow.env')
   # Access environment variables "key" using os.getenv to fetch "value"
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
   # setting up the mlflow regirsty with our tracking uri 
    mlflow.set_registry_uri(MLFLOW_URI)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # os.environ["MLFLOW_TRACKING_URI"]=""
    # os.environ["MLFLOW_TRACKING_USERNAME"]=""
    # os.environ["MLFLOW_TRACKING_PASSWORD"]=""
    app.run(debug=True, port=5001)  ## while deploying or hosting to cloud aws etc, we need to remove (debug=True)

"""
if u want to set port and host to a specfic valid custom setting u can do that like this in __main__ method

if __name__ == "__main__":
    port = 5000
    host = '0.0.0.0'
    app.run(debug=True, host=host, port=port)

"""


### TESTING CODE FOR MLFLOW_MODEL_REGISTRY



# if __name__ == "__main__":
#     # Execute the main program

#     load_dotenv(r"C:\\data science\\Internship projects\\credit card defaulters\\Credit_card_default_prediction_with_mlflow\\mlflow.env")
#      # Access environment variables "key" using os.getenv to fetch "value"
#     MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
#      # setting up the mlflow regirsty with our tracking uri 
#     mlflow.set_registry_uri(MLFLOW_URI)
#     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#     print(f"tracking_url_type_store in __main__: {tracking_url_type_store}")

#     file_path_model_factory = r"C:\\data science\\Internship projects\\credit card defaulters\\Credit_card_default_prediction_with_mlflow\\model_objects\\LogisticRegression(C=10.0).joblib"
                    
#                 # df= pd.read_csv(r"credit_card.csv", )                                           ## using dill library to load file
#     with open(file_path_model_factory, "rb") as file_obj:
#                         model= joblib.load(file_obj)
   
#     with mlflow.start_run():
#           mlflow.sklearn.log_model(model,"model", registered_model_name="BESTModelinstance")
#     # ob = mlflow_demo()
#     # ob.log_mlfow_model()
