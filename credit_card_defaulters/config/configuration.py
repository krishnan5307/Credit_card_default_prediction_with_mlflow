
from credit_card_defaulters.entity.config_entity import DataIngestionConfig, DataTransformationConfig,DataValidationConfig,   \
ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig,TrainingPipelineConfig, ModelFactoryConfig
from credit_card_defaulters.util.util import read_yaml_file
from credit_card_defaulters.logger import logging
import sys,os
from credit_card_defaulters.constant import *
from credit_card_defaulters.exception import CreditException

## COnfiguartion file defines which all files are present in where and it makes link btw constants, yaml file and creates paths
## which can be used in other processess to in pipeline
## This configuartion defnintion class will pull all the data from consatnts and config.yaml file \
## and create paths,ect and returns the code definition of parameters of the entity or interface 
## class defintions for configution


class Configuartion:                  ## The strucutre definition of entity class ( or interface class)

    def __init__(self,
        config_file_path:str =CONFIG_FILE_PATH, ## config.yaml
        current_time_stamp:str = CURRENT_TIME_STAMP
        ) -> None:
        try:
            self.config_info  = read_yaml_file(file_path=config_file_path)         ## All config_yaml file config details
            self.training_pipeline_config = self.get_training_pipeline_config()  ## defined at last of this code
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise CreditException(e,sys) from e


    def get_data_ingestion_config(self) ->DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir           # extracting path from pipeline config to create branch for ingestion
            data_ingestion_artifact_dir=os.path.join(
                artifact_dir,                                                     ## creating ingestion dir inside pipeline
                DATA_INGESTION_ARTIFACT_DIR,
                self.time_stamp
            )
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]              ## using constant folder
            
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]

            tgz_download_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY]
            )

            """
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
            data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
            )
            """
            ingested_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            )
            ingested_train_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
            )
            ingested_test_dir =os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
            )


            data_ingestion_config=DataIngestionConfig(
                dataset_download_url=dataset_download_url, 
                tgz_download_dir=tgz_download_dir, 
                ingested_train_dir=ingested_train_dir, 
                ingested_test_dir=ingested_test_dir
            )
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir       # extracting path from pipeline config to create branch for validation

            data_validation_artifact_dir=os.path.join(
                artifact_dir,                                                       ## creating validation dir inside pipeline
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.time_stamp
            )

            ## eg: data_validation_artifact_dir = "credit_card_defaulters\artifact\data_validation\22-07-2023"

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY] ## accessing schema.yaml file 


            schema_file_path = os.path.join(ROOT_DIR,
            data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
            data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]  ## accessing schema.yaml file 
            )

            report_file_path = os.path.join(data_validation_artifact_dir,
            data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )

            report_page_file_path = os.path.join(data_validation_artifact_dir,
            data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY] ## report.html

            )  

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path, ## where/path to access schema
                report_file_path=report_file_path,  #3 where/path to dump report.json
                report_page_file_path=report_page_file_path,  ## where/path to dump report.html
            ) 
            return data_validation_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transformation_artifact_dir=os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.time_stamp
            )

            data_transformation_config_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            ## add_bedroom_per_room=data_transformation_config_info[DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]


            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )  ## ../preprocessed/preprocessed.pkl

            
            transformed_train_dir=os.path.join(
            data_transformation_artifact_dir,
            data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
            data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            ) ##../transformed_data/train


            transformed_test_dir = os.path.join(
            data_transformation_artifact_dir,
            data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
            data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]

            ) ##../transformed_data/test
            

            data_transformation_config=DataTransformationConfig(  ## add_bedroom_per_room = add_bedroom_per_room,
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir
            )

            logging.info(f"Data transformation config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e: 
            raise CreditException(e,sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_trainer_artifact_dir=os.path.join( 
                artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                self.time_stamp

            )
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
        
            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
            model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],             ## for pickle file 
            model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
            )

            model_config_file_path = os.path.join(model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],   ## for model.yaml file
            model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            )

            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]

            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,     
                base_accuracy=base_accuracy,
                model_config_file_path=model_config_file_path
            )
            logging.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise CreditException(e,sys) from e

    def get_model_evaluation_config(self) ->ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR, )

            model_evaluation_file_path = os.path.join(artifact_dir,
                                                    model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                            time_stamp=self.time_stamp)
            
            
            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise CreditException(e,sys) from e


    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)
             ## eg: C:\Users\krish\credit_card_defaulters\saved_models\20230203141600
            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config

        except Exception as e:
            raise CreditException(e,sys) from e

    def get_training_pipeline_config(self) ->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
            training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
            training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]  ## key value pairs from abov dict
            )

            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            ## eg: TrainingPipelineConfig(artifact_dir="ROOT_DIR\credit_Card_defaulters\artifact")
            logging.info(f"Training pipleine config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise CreditException(e,sys) from e
        
    def get_model_factory_config(self) -> ModelFactoryConfig:
        try:
            model_factory_config = self.config_info[MODEL_FACTORY_CONFIG_KEY]
            mlflow_uri = model_factory_config[MODEL_FACTORY_MLFLOW_URI_KEY]
            model_factory_config = ModelFactoryConfig(mlflow_uri=mlflow_uri)
            logging.info(F"model factory config: {model_factory_config}")
            return model_factory_config
        except Exception as e:
            raise CreditException(e,sys) from e 