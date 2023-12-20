

import html
from credit_card_defaulters.logger import logging
from credit_card_defaulters.exception import CreditException
from credit_card_defaulters.entity.config_entity import DataValidationConfig ## to give paramters to class, 
## we fetch this userdefined datatype from entity or interface class
from credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys,yaml
import pandas  as pd
from sklearn import datasets, ensemble, model_selection

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import RegressionPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import TargetDriftPreset
from credit_card_defaulters.constant import *
from credit_card_defaulters.util.util import read_yaml_file
import json

class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,  ## we recive paramerters through 
                 ##configuration.py  when we call this class in pipeline 
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CreditException(e,sys) from e


    def get_train_and_test_df(self): ## used in get_and_save_data_drift_report and save_data_drift_report_page() 
                                     ## function defined below
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise CreditException(e,sys) from e


    def is_train_test_file_exists(self)->bool:
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False 

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available =  is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")
            
            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file}" \
                    "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise CreditException(e,sys) from e

    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False

            import pandas as pd
            train_df,test_df = self.get_train_and_test_df()

          # Define the expected schema
            expected_schema_path = self.data_validation_config.schema_file_path
            schema_yaml  = read_yaml_file(file_path=expected_schema_path)
            expected_schema = schema_yaml['columns']

            original_train = train_df.dtypes.to_dict() ## converting from df to dict
            original_test = test_df.dtypes.to_dict()
            ##assert train_df.dtypes.to_dict() == expected_schema

            # Remove "dtype" and parentheses from the values
            original_train_final = {key: str(value).replace('dtype(', '').replace(')', '') for key, value in original_train.items()}
            original_test_final = {key: str(value).replace('dtype(', '').replace(')', '') for key, value in original_test.items()}

                        
            # Compare dictionaries
            try:
                if original_train_final == expected_schema and original_test_final == expected_schema:
                    print("Schema Validation Successsfull")
                    logging.info("Schema Validation Successsfull:")
                    validation_status = True
            except AssertionError:
                logging.info("Schema Validation failed:")  
            
            return validation_status 
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_and_save_data_drift_report(self): ## for data_driftprofile and saving report
        try:


            """
            column mapping example- needed as it defines structure/schema to the evidently data validations

            data = {
                    'target': [1, 0, 1, 0, 1],
                    'prediction': [0.9, 0.2, 0.8, 0.1, 0.7],
                    'numerical_feature_1': [10, 15, 20, 25, 30],
                    'numerical_feature_2': [0.5, 1.2, 0.8, 1.5, 1.0],
                    'categorical_feature': ['A', 'B', 'A', 'C', 'B']
                }
            
                # Assuming you have numerical_features and categorical_features defined

                    numerical_features = ['numerical_feature_1', 'numerical_feature_2']
                    categorical_features = ['categorical_feature']

                    column_mapping = {
                        'target': 'target',
                        'prediction': 'prediction',
                        'numerical_features': numerical_features,
                        'categorical_features': categorical_features
                }


                regression_performance_report = Report(metrics=[
                    RegressionPreset(),
                ])
           ## added column mapping below

                regression_performance_report.run(reference_data=target_col, current_data=prediction_col,
                 column_mapping=column_mapping)

                regression_performance_report

            """

                        # profile = Profile(sections=[DataDriftProfileSection()])
                        # profile.calculate(train_df,test_df)
                        # report = json.loads(profile.json())

            strat_train_df,strat_test_df = self.get_train_and_test_df()
            ## PERFORMING Evidently preset operations
        
           ## DATA drift preset

            data_drift_report = Report(metrics=[DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),])

            data_drift_report.run(reference_data=strat_train_df, current_data=strat_test_df)

            ## DATA quality preset

            data_quality_report = Report(metrics=[DataQualityPreset(),])

            data_quality_report.run(reference_data=strat_train_df, current_data=strat_test_df)
            
            
            report_file_path = self.data_validation_config.report_file_path ## getting path only
            report_dir = os.path.dirname(report_file_path)    ## creating dir now if it doesnt exists
            os.makedirs(report_dir,exist_ok=True)

            data_drift_report.save_json(report_file_path)
            data_quality_report.save_json(report_file_path)

            """
             "IMPORTANT"

            report_dir = os.path.dirname(report_file_path):
             os.path.dirname(report_file_path): This extracts the directory path from the given report_file_path. 
             For example, if report_file_path is '/path/to/report/report.txt', report_dir will be '/path/to/report'.

            os.makedirs(report_dir, exist_ok=True):
             os.makedirs(report_dir): This attempts to create the directory specified by report_dir.
             exist_ok=True: This argument ensures that the function does not raise an error if the directory already exists. 
             If exist_ok is set to False (or not provided), and the directory already exists, it would raise a 
             FileExistsError.
            
            """

            # with open(report_file_path,"w") as report_file:
            #     json.dump(data_drift_report, report_file, indent=6) ## indent=6 means that each level of nesting in the JSON file will be 
            #     json.dump(data_quality_report, report_file, indent=6)
            #     ## indented with 6 spaces from parenthesis.
            #     """   eg:  {------
            #                       abjb: cnknc
            #                       cbfnbf:vnfkvn
            #                }
            #     """     
            return data_drift_report, data_quality_report
        except Exception as e:
            raise CreditException(e,sys) from e
            print(e)

    def save_data_drift_report_page(self): ## for dashboard and saving report page.html
        try:
            # dashboard = Dashboard(tabs=[DataDriftTab()])
            # strat_train_df,strat_test_df = self.get_train_and_test_df()
            # dashboard.calculate(train_df,test_df)


            strat_train_df,strat_test_df = self.get_train_and_test_df()
        
           ## DATA drift preset page html

            data_drift_report = Report(metrics=[DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),])

            data_drift_report.run(reference_data=strat_train_df, current_data=strat_test_df)

            ## DATA quality preset page html

            data_quality_report = Report(metrics=[DataQualityPreset(),])

            data_quality_report.run(reference_data=strat_train_df, current_data=strat_test_df)
 


            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path) ##passing path to fetch dir name
            ## cc//vvv//bb.json - then cc//vvv is the directory name
            os.makedirs(report_page_dir,exist_ok=True)

            data_drift_report.save_html(report_page_file_path)
            data_quality_report.save_html(report_page_file_path)

            # with open(report_page_file_path,"w") as report_file:
            #     html.dump(data_drift_report, report_file, indent=6) ## indent=6 means that each level of nesting in the JSON file will be 
            #     html.dump(data_quality_report, report_file, indent=6)

            # dashboard.save(report_page_file_path)
        except Exception as e:
            raise CreditException(e,sys) from e

    def is_data_drift_found(self)->bool:
        try:
            report_json,quality_json = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise CreditException(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e


    def __del__(self): ## calling destructor when object of class is terminated
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
        



