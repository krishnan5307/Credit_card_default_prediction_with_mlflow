from insurance.entity.config_entity import DataIngestionConfig
import sys,os
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity.artifact_entity import DataIngestionArtifact
## import tarfile
import numpy as np
## import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from insurance.constant import *
from config import configuration

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):  ## constructor assigned with parameter as data_ingestion_config while calling class DataIngestion from function in pipeline
        try:                                                        # ie we pass congif info while calling this class DataIngestion in pipleine
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
           ## self.database_connection = connect_database()
            self.database_connection = configuration()
        except Exception as e:
            raise InsuranceException(e,sys)
    

    def download_insurance_data(self,) -> str:
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

            


            # df = pd.DataFrame()
            # df = pd.read_csv("dataset/insurance.csv")
            

            #df = pd.read_csv(r"insurance\dataset\dataset.csv",delimiter=",")


            #folder location to download file
        

            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            os.makedirs(tgz_download_dir,exist_ok=True)

            insurance_file_name = download_url

            tgz_file_path = os.path.join(tgz_download_dir, insurance_file_name)

            logging.info(f"Downloading file from :[{self.database_connection}] into :[{tgz_file_path}]")

            df.to_csv(tgz_file_path, mode="w", index=False, header=True)
            ## urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path

        except Exception as e:
            print(e)
            raise InsuranceException(e,sys) from e

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

            file_name = os.listdir(tgz_download_dir)[0]    ## we are taking/copying file name inside a folder raw_data_dir (1st index of that list output)

            ##insurance_file_path = os.path.join(tgz_download_dir,file_name)  ## creating file path

            insurance_file_path = tgz_file_path     

            logging.info(f"Reading csv file: [{insurance_file_path}]")
            insurance_data_frame = pd.read_csv(insurance_file_path)       ## reading file path

            insurance_data_frame["premium_cat"] = pd.cut(           
                insurance_data_frame["expenses"],       ## stratified state means distribution of test and train dataset should align during splits
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],    ## creating stratified split
                labels=[1,2,3,4,5]
            )
            

            logging.info(f"Splitting data into train and test")
            strat_train_set = None                        ## stratified state means distribution of test and train dataset should align during splits
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in split.split(insurance_data_frame, insurance_data_frame["premium_cat"]):
                strat_train_set = insurance_data_frame.loc[train_index].drop(["premium_cat"],axis=1)
                strat_test_set = insurance_data_frame.loc[test_index].drop(["premium_cat"],axis=1)

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
            raise InsuranceException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_insurance_data()
            ##self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test(tgz_file_path)
        except Exception as e:
            raise InsuranceException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
