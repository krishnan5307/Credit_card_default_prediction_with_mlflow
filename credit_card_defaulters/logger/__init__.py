import logging
from datetime import datetime
import os, sys
import pandas as pd
from credit_card_defaulters.constant import get_current_time_stamp 
LOG_DIR="logs"

def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"

LOG_FILE_NAME=get_log_file_name()

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

## defining the logging

logging.basicConfig(#filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
level=logging.INFO
)


# Create a FileHandler and add it to the root logger
file_handler = logging.FileHandler(LOG_FILE_PATH)  ## defining handler to a variable
formatter = logging.Formatter('[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s')
file_handler.setFormatter(formatter)  ## passing loggin format to filehandler using setFormatter()
logging.getLogger().addHandler(file_handler)  # adding this filehandler to logging 

# Create a StreamHandler and add it to the root logger
stream_handler = logging.StreamHandler(sys.stdout) ## defining handler to a variable
stream_handler.setFormatter(formatter) ## passing loggin format to Streamhandler using setFormatter()
logging.getLogger().addHandler(stream_handler) # adding this streamhandler to logging 


"""

If you want to add multiple handlers to the logging configuration, including both a FileHandler for writing to a file and a StreamHandler for printing log messages to the console, you can modify the code as follows:

python
Copy code
import logging
import sys

LOG_FILE_PATH = "example.log"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),           # FileHandler for writing to a file
        logging.StreamHandler(sys.stdout)             # StreamHandler for printing to the console
    ],
    level=logging.INFO
)
Explanation:

handlers parameter:

The handlers parameter is now a list containing both a FileHandler and a StreamHandler.
logging.FileHandler(LOG_FILE_PATH) adds a handler that writes log messages to the specified file (example.log).
logging.StreamHandler(sys.stdout) adds a handler that writes log messages to the console (stdout).
StreamHandler(sys.stdout) for console output:

The StreamHandler is used for printing log messages to the console (sys.stdout). You can customize this part based on where you want to output log messages.
"""

## if u need to get dataframe of log
def get_log_dataframe(file_path) -> pd.DataFrame:
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():   
            data.append(line.split("^;")) ## splitting at this level

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Log Level","line number","file name","function name","message"] ## same as format. line21
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["message"]

    return log_df[["log_message"]]


""""

The logging methods are ordered by increasing severity from bottom to top. 
The order, from least severe to most severe, is typically as follows:

DEBUG: Used for detailed debugging information.
INFO: Used to confirm that things are working as expected.
WARNING: Indicates a potential problem or a situation that might cause issues in the future.
ERROR: Indicates a more serious problem that prevented the application from performing a specific function.
CRITICAL: Indicates a very serious error, often resulting in a complete failure of the application.
When configuring logging, you can set the logging level to control which messages are actually logged. 
For example, setting the logging level to INFO will log messages at the INFO, WARNING, ERROR, and CRITICAL levels, 
but not DEBUG messages. It's a way to control the verbosity of the log output based on the severity of the 
messages.

"""