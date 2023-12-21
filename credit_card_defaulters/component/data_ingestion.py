from credit_card_defaulters.entity.config_entity import DataIngestionConfig
import sys,os
from credit_card_defaulters.exception import CreditException
from credit_card_defaulters.logger import logging
from credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact
## import tarfile
import numpy as np
## import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from credit_card_defaulters.constant import *
from config import configuration

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):  ## constructor assigned with parameter as data_ingestion_config while calling class DataIngestion from function in pipeline
        try:                                                        # ie we pass congif info while calling this class DataIngestion in pipleine
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
           ## self.database_connection = connect_database()
            self.database_connection = configuration()
        except Exception as e:
            raise CreditException(e,sys)
    

    def download_credit_data(self,) -> str:
        try:

            #from cassandra database using connction variable to download dataset to dataframe and later to csv
            download_url = self.data_ingestion_config.dataset_download_url
            # logging.info(f"connction sent to connect_databsae.py file")

            # session = self.database_connection.get_db_connection()
            
            # sql_query = "SELECT * FROM {}.{};".format(constant.CASSANDRA_KEYSPACE, constant.CASSANDRA_TABLE)
            # df = pd.DataFrame()
            # for row in session.execute(sql_query):
            #         df = df.append(pd.DataFrame(row, index=[0]))

            # session.shutdown()

            # df = df.reset_index(drop=True).fillna(pd.np.nan)
            #df = configuration.start()

            df = self.database_connection.get_configuration()
            ## removing first ID column now itself
            df.drop("ID", axis=1, inplace=True)

            


            # df = pd.DataFrame()
            # df = pd.read_csv("dataset/insurance.csv")
            

            #df = pd.read_csv(r"insurance\dataset\dataset.csv",delimiter=",")


            #folder location to download file
        

            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            os.makedirs(tgz_download_dir,exist_ok=True)

            credit_file_name = download_url

            tgz_file_path = os.path.join(tgz_download_dir, credit_file_name)

            logging.info(f"Downloading file from :[{self.database_connection}] into :[{tgz_file_path}]")

            df.to_csv(tgz_file_path, mode="w", index=False, header=True)
            ## urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            print("succeffully completed data ingestion: checking for debugging")
            return tgz_file_path

        except Exception as e:
            print(e)
            raise CreditException(e,sys) from e

            """

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            with tarfile.open(tgz_file_path) as housing_tgz_file_obj:
                housing_tgz_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

        except Exception as e:
            raise InsuranceException(e,sys) from e
    
           """

    def split_data_as_train_test(self,tgz_file_path:str) -> DataIngestionArtifact:
        try:
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
        
            ##raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(tgz_download_dir)[0]    
            ### Assuming tgz_download_dir is the directory path
            ### eg: tgz_download_dir = "/path/to/your/directory"
            ##  os.listdir(tgz_download_dir): This function returns a list of the files and folders in the specified directory (tgz_download_dir).
            ## [0]: This index is used to select the first item in the list of files. 
        

            credit_file_path = tgz_file_path     

            logging.info(f"Reading csv file: [{credit_file_path}]")
            credit_data_frame = pd.read_csv(credit_file_path)       ## reading file path
            # ## pyspark reanming target column name to match syntax
            # credit_data_frame = credit_data_frame.withColumnRenamed("default.payment.next.month", "default_payment_next_month")
            ## pandas reanming target column name to match syntax
            credit_data_frame = credit_data_frame.rename(columns={"default.payment.next.month": "default_payment_next_month"})

            # credit_data_frame["credit_cat"] = pd.cut(           
            #     credit_data_frame["default.payment.next.month"],       ## stratified state means distribution of test and train dataset should align during splits
            #     bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],    ## creating stratified split
            #     labels=[1,2,3,4,5]
            # )
            

            logging.info(f"Splitting data into train and test")
            strat_train_set = None                        ## stratified state means distribution of test and train dataset should align during splits
            strat_test_set = None

            spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) ## instantiating splitter

            for train_index,test_index in spliter.split(credit_data_frame, credit_data_frame["default_payment_next_month"]):
                strat_train_set = credit_data_frame.loc[train_index] ##.drop(["premium_cat"],axis=1)
                strat_test_set = credit_data_frame.loc[test_index] ##.drop(["premium_cat"],axis=1)



            """
            -spliter: This is an instance of the StratifiedShuffleSplit class, which is a cross-validator used for creating stratified train-test splits.
            -credit_data_frame: This is the DataFrame that you want to split into training and testing sets.
            -credit_data_frame["premium_cat"]: This is the target variable or a categorical variable used for stratified splitting. 
             The data will be split in a way that ensures a similar distribution of this variable in both the training and testing sets.
            -split(credit_data_frame, credit_data_frame["premium_cat"]): This method generates indices for the train-test split based on the specified data and target variable.
            
            "IMPORTANT"
            This loop only iterates only once, generates 1 pair of train and test index arrays as n_split=1 here
            eg: train_index = [2, 5, 6, 9, 3, 8, 4, 7]
                test_index = [1, 0]

            
            
            """    

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_credit_data()
            ##self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test(tgz_file_path)
        except Exception as e:
            raise CreditException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")


"""

The '__del__' method is a special method in Python classes, also known as the "destructor." 
It is automatically called when an object is about to be destroyed, either because it goes out of scope 
or because it is explicitly deleted. In this case, the __del__ method is used to log a message 
indicating the completion of a data ingestion process.

The purpose of including such a message in the destructor could be to indicate the completion 
of the data ingestion process whenever an instance of the class is destroyed. 
This could be useful for logging or debugging purposes, providing information about the lifecycle of the object.

"""


"""
f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n": This is an f-string, a way to format strings in Python.
 It's used to construct the log message. The message itself is designed to include a visual representation 
 of double arrows ('>>'*20 and '<<'*20) for emphasis.

The resulting log message might look something like this:

>>>>>>>>>>>>>>>>>>>>>>>>>>>>Data Ingestion log completed.<<<<<<<<<<<<<<<<<<<<<<<<<<<<


"""