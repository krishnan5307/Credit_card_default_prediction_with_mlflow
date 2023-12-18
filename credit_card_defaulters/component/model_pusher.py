from credit_card_defaulters.logger import logging
from credit_card_defaulters.exception import CreditException
from credit_card_defaulters.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact 
from credit_card_defaulters.entity.config_entity import ModelPusherConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise CreditException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            ## model trained will be here in this path and if and only if model evaltuon status if true, 
            ## this pusher will execute
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path ## trained model path
            export_dir = self.model_pusher_config.export_dir_path ## saved models folder paath
            model_file_name = os.path.basename(evaluated_model_file_path)## fetching model name to save it into saved models folder
            export_model_file_path = os.path.join(export_dir, model_file_name) ## final path to push model into saved models filder
            logging.info(f"Exporting model file: [{export_model_file_path}]")
            os.makedirs(export_dir, exist_ok=True)
            
            ## now copying the file from source to destination path
            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)
            #we can call a function to save model to push to Azure blob storage/ google cloud strorage / s3 bucket
            logging.info(
                f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")