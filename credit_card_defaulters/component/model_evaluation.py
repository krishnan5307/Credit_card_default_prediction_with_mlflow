

from credit_card_defaulters.logger import logging
from credit_card_defaulters.exception import CreditException
from credit_card_defaulters.entity.config_entity import ModelEvaluationConfig
from credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from credit_card_defaulters.constant import *
import numpy as np
import os
import sys
from credit_card_defaulters.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from credit_card_defaulters.entity.model_factory import evaluate_classification_model, evaluate_regression_model




class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_best_model(self):
        try:

            ## line34-40 : only for first tym if any file doesnt exists
            ## ## if file doesnt exist we create empty model evalution.yaml file
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):                
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model                ## return no model- None and exit this funtion
            
            
            ## reading the content of model evaluation.yaml
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            ## if file is there but best model key is none we return no model
            if BEST_MODEL_KEY not in model_eval_file_content:      
                return model            ## return no model- None  and exit this funtion
            
            
            ## if best model avilable in BEST_MODEL_KEY
            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])     
            return model
        except Exception as e:
            raise CreditException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):  ## will be called only if trained model is better than base model
        try:
            ## reading model evalution.yaml  file
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path      ## base model path in prdction
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            

            eval_result = {    ## new model info to be added
                BEST_MODEL_KEY: {                                                     
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }
            previous_best_model = None

            ## if previous_best_model  exists in production                           
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]  
            logging.info(f"Previous eval result: {model_eval_content}")         
            if previous_best_model is not None:
                ## trying to prepare history dict data of previous model exists to upadte ith with new data
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model} 
                if HISTORY_KEY not in model_eval_content:    ## if no histry key  
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history) ## appending the new histry key to newmodel info eval_result dict to be added
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:                     ## retraining model
        try:
            ## getting trained model object
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            ## loading DataFrame
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column for train and test df
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            ## getting best model from model evaluation.yaml file or it can be empty if no trained model for first tym
            model = self.get_best_model() 
            

            ## for first time if there is no model in model evaluation.yaml file
            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                ## now we update the model evaluation  .yaml file with model evaluation artifact
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            ## SO if there was no model earlier in production we simply add to eval report and return model_evaluation_artifact



            ## Now if there was no model earlier in production we need to evaluate new model with 
            ## previous best model in prodcution - evalutiton
            model_list = [model, trained_model_object]
            ## now we do the same model evaltuion we did during model trainig with bunch of model as list
            ## ## gives best model of 2 in model_list
            metric_info_artifact = evaluate_classification_model(model_list=model_list,      
                                                               X_train=train_dataframe,
                                                               y_train=train_target_arr,
                                                               X_test=test_dataframe,
                                                               y_test=test_target_arr,  
                                                               base_accuracy=self.model_trainer_artifact.model_f1_score,
                                                               )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None: ## No trained model with acceptable acuray found, to be added to predcution
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1: ## index of trained_model_object in model_list
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            print(e)
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")