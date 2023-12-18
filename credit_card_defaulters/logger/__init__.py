import logging
from datetime import datetime
import os
import pandas as pd
from credit_card_defaulters.constant import get_current_time_stamp 
LOG_DIR="logs"

def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"

LOG_FILE_NAME=get_log_file_name()

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)



logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
level=logging.INFO
)
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