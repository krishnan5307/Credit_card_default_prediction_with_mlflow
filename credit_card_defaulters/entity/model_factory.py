from cmath import log
import importlib
from pyexpat import model
import numpy as np
import pandas as pd
import yaml
from credit_card_defaulters.exception import CreditException
import os
import sys, joblib
import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
from collections import namedtuple
from typing import List
from credit_card_defaulters.logger import logging
from sklearn.metrics import r2_score,mean_squared_error
from credit_card_defaulters.constant import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
# MLFLOW_TRACKING_URI= "https://dagshub.com/krishnan5307/Credit_card_default_prediction_with_mlflow.mlflow"
# mlflow.set_registry_uri(MLFLOW_TRACKING_URI)


# import config_entity

tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])
## for regression
# MetricInfoArtifact = namedtuple("MetricInfoArtifact",
#                                 ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
#                                  "test_accuracy", "model_accuracy", "index_number"])
## for classification

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object",
                                 "train_precision", "test_precision",
                                 "train_recall", "test_recall",
                                 "train_f1_score", "test_f1_score",
                                 "model_f1_score",
                                 "train_accuracy", "test_accuracy",
                                 "auc_roc", "auc_value",
                                 "index_number"])




def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6)->MetricInfoArtifact:
    pass


def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
    """

    we only recieved the latest model with previously best trained models in model_list, so now we need to
    evaluate the model and find the metrics like rmse etc

    Description:
    This function compare multiple regression model return best model

    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature

    return
    It retured a named tuple
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

    """



    """
    for model_instance in model_list:
    if isinstance(model_instance, GridSearchedBestModel):
        print("model_instance is an instance of GridSearchedBestModel")
    else:
        print("model_instance is not an instance of GridSearchedBestModel")
    """
    try:     
        
        ## for model training where we pass the original GridSearchedBestModel list which contain best_estimator, best_score,best_params etc
        if all(isinstance(item, GridSearchedBestModel) for item in model_list):
            logging.info(f"{'>>'*30}Started GridSearchedBestModel list training: [{model_list}] {'<<'*30}")

                        
            
            with mlflow.start_run(run_name="All Trained Models"):
                try:
                    
                    model = None
                    index_number = 0 ## for model index
                    metric_info_artifact = None
                    for mod in model_list: 
                        model = mod.best_model
                        model_name = str(model)  #getting model name based on model object
                        logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        # ## To chcek unique values in predictions
                        # print(y_train_pred)
                        # # Convert to DataFrame
                        # df_from_array = pd.DataFrame(y_train_pred, columns=['prediction'])
                        # distinct_values = df_from_array['prediction'].unique()
                        # print(distinct_values)


                        # Calculating precision, recall, and F1 score on training and testing dataset
                        train_precision = precision_score(y_train, y_train_pred, average='weighted')
                        test_precision = precision_score(y_test, y_test_pred, average='weighted')

                        train_recall = recall_score(y_train, y_train_pred, average='weighted')
                        test_recall = recall_score(y_test, y_test_pred, average='weighted')

                        train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
                        test_f1_score = f1_score(y_test, y_test_pred, average='weighted')

                        # Calculating accuracy on training and testing dataset
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        test_accuracy = accuracy_score(y_test, y_test_pred)

                        # Calculating harmonic mean of train_f1_score and test_f1_score
                        model_f1_score = (2 * (train_f1_score * test_f1_score)) / (train_f1_score + test_f1_score)

                        # Calculating AUC-ROC and ROC score
                        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
                        roc_auc = auc(fpr, tpr)
                        auc_value = roc_auc_score(y_test, y_test_pred)

                        # Calculate the absolute difference between test_accuracy and train_accuracy
                        diff_test_train_acc = abs(test_accuracy - train_accuracy)            


                        
                        #logging all important metric
                        logging.info(f"{'>>'*30} Score {'<<'*30}")
                        logging.info(f"Train acc Score\t\t Test acc Score\t\t model avg Score")
                        logging.info(f"{train_accuracy}\t\t {test_accuracy}\t\t{model_f1_score}")

                        logging.info(f"{'>>'*30} Loss {'<<'*30}")
                        logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
                        logging.info(f"Train precision: [{train_precision}].")
                        logging.info(f"Test precision: [{test_precision}].")
                        logging.info(f"Train recall: [{train_recall}].")
                        logging.info(f"Test recall: [{test_recall}].")
                        logging.info(f"Train f1_score: [{train_f1_score}].")
                        logging.info(f"model_f1_score: [{model_f1_score}].")
                        logging.info(f"auc_roc: [{roc_auc}].")
                        logging.info(f"auc_value: [{auc_value}].")
                        
                        print(model)
                        # Save the trained model locally
                        model_obj_path = "C:\\data science\\Internship projects\\credit card defaulters\\Credit_card_default_prediction_with_mlflow\\model_objects"
                        # Your directory path
                       # creating file path
                        file_path = os.path.join(model_obj_path, f"{model_name}.joblib")
                        with open(file_path, 'wb') as file:
                            joblib.dump(model, file)
                        model_path = file_path   
                        loaded_model = joblib.load(model_path)
                        # Confirm the loaded_model type
                        print(type(loaded_model)) 

                       ## adding mlflow tracking service for parameters snd models

                        with mlflow.start_run(run_name=model_name, nested=True):
                            try:
                                    #  mlflow.log_params() 

                                mlflow.log_metric("Diff test train accuracy", diff_test_train_acc)
                                mlflow.log_metric("Train precision", train_precision)
                                mlflow.log_metric("Test precision", test_precision)
                                mlflow.log_metric("Train recall", train_recall)
                                mlflow.log_metric("Test recall", test_recall)
                                mlflow.log_metric("model_f1_score", model_f1_score)
                                mlflow.log_metric("auc_value", auc_value)
                                mlflow.log_metric("auc_roc", roc_auc)

                                

                                #  mlflow.sklearn.log_model( "model", mod.model)
                                mlflow.log_metric("model_best_score", mod.best_score)
                                mlflow.log_param("model_serial_number", mod.model_serial_number)
                                mlflow.log_params(mod.best_parameters)

                                if tracking_url_type_store != "file":
                                    print(f"tracking_url_type_store in training call: {tracking_url_type_store}")
                                    # Register the model
                                    # There are other ways to use the Model Registry, which depends on the use case,
                                    # please refer to the doc for more information:
                                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                                    mlflow.sklearn.log_model(model, model_name, registered_model_name= model_name)
                                else:
                                    mlflow.sklearn.log_model(model, "model")                                    
                            finally:
                                pass
                                # mlflow.end_run()
                        ## Two criterias to accept model
                        #if model accuracy is greater than base accuracy and train and test score is within certain thershold
                        #we will accept that model as accepted model
                        if model_f1_score >= base_accuracy and diff_test_train_acc < 0.05 and auc_value >0:
                            base_accuracy = model_f1_score
                            metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                                    model_object=model,
                                                                    train_precision=train_precision,
                                                                    test_precision=test_precision,
                                                                    train_recall=train_recall,
                                                                    test_recall=test_recall,
                                                                    train_f1_score=train_f1_score,   
                                                                    test_f1_score=test_f1_score, 
                                                                    model_f1_score =model_f1_score,
                                                                    train_accuracy=train_accuracy,
                                                                    test_accuracy=test_accuracy,
                                                                    auc_roc= roc_auc, auc_value=auc_value,                                                    
                                                                    index_number=index_number)
                            ## index of model added if satisy the criteria
                            logging.info(f"Acceptable model found {metric_info_artifact}. ")
                        index_number += 1  ## index of each model to add it in as model index if satisy the criteria
                    if metric_info_artifact is None:
                        logging.info(f"No model found with higher accuracy than base accuracy")
                    return metric_info_artifact  ## will return only one model with these two conditions base_acc and diff_test_train_acc
                except Exception as e:
                    print(e)
                    raise CreditException(e, sys) from e



                # finally:
                #     mlflow.end_run()





        else:
            
            ## for model evaluation where pass only list of models of type gridSearchBestmodel.best_model
            with mlflow.start_run(run_name="Latest Evaluated models to be selected to production"):
                try:
                    model = None
                    index_number = 0 ## for model index
                    metric_info_artifact = None
                    for model in model_list: ## .pkl file models loaded from trained models folder,(current model and previous best model )
                        
                        model_name = str(model)  #getting model name based on model object
                        logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        # ## To chcek unique values in predictions
                        # print(y_train_pred)
                        # # Convert to DataFrame
                        # df_from_array = pd.DataFrame(y_train_pred, columns=['prediction'])
                        # distinct_values = df_from_array['prediction'].unique()
                        # print(distinct_values)


                        # Calculating precision, recall, and F1 score on training and testing dataset
                        train_precision = precision_score(y_train, y_train_pred, average='weighted')
                        test_precision = precision_score(y_test, y_test_pred, average='weighted')

                        train_recall = recall_score(y_train, y_train_pred, average='weighted')
                        test_recall = recall_score(y_test, y_test_pred, average='weighted')

                        train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
                        test_f1_score = f1_score(y_test, y_test_pred, average='weighted')

                        # Calculating accuracy on training and testing dataset
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        test_accuracy = accuracy_score(y_test, y_test_pred)

                        # Calculating harmonic mean of train_f1_score and test_f1_score
                        model_f1_score = (2 * (train_f1_score * test_f1_score)) / (train_f1_score + test_f1_score)

                        # Calculating AUC-ROC and ROC score
                        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
                        roc_auc = auc(fpr, tpr)
                        auc_value = roc_auc_score(y_test, y_test_pred)

                        # Calculate the absolute difference between test_accuracy and train_accuracy
                        diff_test_train_acc = abs(test_accuracy - train_accuracy)            


                        
                        #logging all important metric
                        logging.info(f"{'>>'*30} Score {'<<'*30}")
                        logging.info(f"Train acc Score\t\t Test acc Score\t\t model avg Score")
                        logging.info(f"{train_accuracy}\t\t {test_accuracy}\t\t{model_f1_score}")

                        logging.info(f"{'>>'*30} Loss {'<<'*30}")
                        logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
                        logging.info(f"Train precision: [{train_precision}].")
                        logging.info(f"Test precision: [{test_precision}].")
                        logging.info(f"Train recall: [{train_recall}].")
                        logging.info(f"Test recall: [{test_recall}].")
                        logging.info(f"Train f1_score: [{train_f1_score}].")
                        logging.info(f"model_f1_score: [{model_f1_score}].")
                        logging.info(f"auc_roc: [{roc_auc}].")
                        logging.info(f"auc_value: [{auc_value}].")

                        ## adding mlflow tracking service for parameters snd models

                        with mlflow.start_run(run_name=model_name, nested=True):
                            try:
                        # #             #  mlflow.log_params() 

                                mlflow.log_metric("Diff test train accuracy", diff_test_train_acc)
                                mlflow.log_metric("Train precision", train_precision)
                                mlflow.log_metric("Test precision", test_precision)
                                mlflow.log_metric("Train recall", train_recall)
                                mlflow.log_metric("Test recall", test_recall)
                                mlflow.log_metric("model_f1_score", model_f1_score)
                                mlflow.log_metric("auc_value", auc_value)
                                mlflow.log_metric("auc_roc", roc_auc)
## we cannot log params in mlflow during model evaluaion stage as model is saved in training artifacts as gridsearchbestmodel.best_model
# or in oother words gridsearchcv.best_estimator_ . so we load the model only and not gridsearchbestmodel.
# so we dont have gridsearchbestmodel.best_params or best_score in this stage, but it was available in model traiing stage which is above code till line 215
# where we used [gridsearchbestmodel] list of all initalised models 
                              #  mlflow.sklearn.log_model(model, "model")
                                # mlflow.log_metrics(model.best_score)
                                # mlflow.log_param("model_serial_number", model.model_serial_number)
                            # #     # mlflow.log_params(model.best_parameters)
                                
                                if tracking_url_type_store != "file":
                                    print(f"tracking_url_type_store in model evalution call : {tracking_url_type_store}")

                                    # Register the model
                                    # There are other ways to use the Model Registry, which depends on the use case,
                                    # please refer to the doc for more information:
                                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                                    mlflow.sklearn.log_model(model, model_name, registered_model_name= model_name)
                                else:
                                    mlflow.sklearn.log_model(model, "model")                                 
                            finally:
                                pass
                                # mlflow.end_run()
                        ## Two criterias to accept model
                        #if model accuracy is greater than base accuracy and train and test score is within certain thershold
                        #we will accept that model as accepted model
                        if model_f1_score >= base_accuracy and diff_test_train_acc < 0.05 and auc_value >0:
                            base_accuracy = model_f1_score
                            metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                                    model_object=model,
                                                                    train_precision=train_precision,
                                                                    test_precision=test_precision,
                                                                    train_recall=train_recall,
                                                                    test_recall=test_recall,
                                                                    train_f1_score=train_f1_score,   
                                                                    test_f1_score=test_f1_score, 
                                                                    model_f1_score =model_f1_score,
                                                                    train_accuracy=train_accuracy,
                                                                    test_accuracy=test_accuracy,
                                                                    auc_roc= roc_auc, auc_value=auc_value,                                                    
                                                                    index_number=index_number)
                            ## index of model added if satisy the criteria
                            logging.info(f"Acceptable model found {metric_info_artifact}. ")
                        index_number += 1  ## index of each model to add it in as model index if satisy the criteria
                    if metric_info_artifact is None:
                        logging.info(f"No model found with higher accuracy than base accuracy")
                    return metric_info_artifact  ## will return only one model with these two conditions base_acc and diff_test_train_acc
                except Exception as e:
                    print(e)
                    raise CreditException(e, sys) from e

    except Exception as e:
        raise CreditException(e,sys) from e
                    # finally:
                    #     mlflow.end_run()             
      

      
            # Save model
            #    mlflow.sklearn.log_model(initialized_model, "model")
                # Log model signature
                #signature = infer_signature(initialized_model.model)
            #   mlflow.sklearn.log_model(initialized_model.model, "model", signature=signature)


            # Save custom artifact
            #  artifact_path = "custom_metrics"
            #  mlflow.log_artifact(str(metric_info_artifact), artifact_path)
 




def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },  
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise CreditException(e, sys)


class ModelFactory:
    def __init__(self, model_config_path: str = None,):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY] #GridSearchCV
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY] #sklearn.model_selection
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY]) ## cv: 5
                                                                                                 ## verbose: 2

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY]) ## all models info 

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod ## updating model with params passes as dict
    def update_property_of_class(instance_ref:object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value) ## setattr(class,'attribute in string', value) ie, in the clas it set attribute value to given vlaue
            return instance_ref
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):      ## for importung the librarbies of modeule and model
        try:
            ## load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            ## get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)  ## returns eg : LogisticRegression from sklearn.linear_model
            ## This would return the 'LogisticRegression' class from the scikit-learn module.
            return class_ref                            
        except Exception as e:
            raise CreditException(e, sys) from e 

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class            
            
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module, ## grid search CV variable
                                                             class_name=self.grid_search_class_name
                                                             )## gives GridSearchCV
            ## final GridSearchCV with parameters like model and params_grids
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,       ## passing and initializing parameters of grid_search_cv to its object grid_search_cv_ref
                                                param_grid=initialized_model.param_grid_search)
            ## updated final GridSearchCV with parameters
              # we call model as estimator in gridSearchCV 
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data) ## update params accoding to model.yaml file
                     ## eg: {params: CV=5 , verbose=2} for gird_search_cv before fitting with model and its params
            
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            ## training model in gridsearch
            grid_search_cv.fit(input_feature, output_feature) ## training the model 
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_, ## final model best model
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
            )
            # we call model as estimator in gridSearchCV                                                 
            # using mlflow tracking serivve for all models and its best params from GridCV
        

            

            return grid_searched_best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:   ## each model in model config yaml is updated with imported lib, updated with params
                                            ## returns list of namedtuples of type InitializedModelDetail
        """
        This function will return a list of model details.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():  ## model seriel no: means  model 0, model 1 etc
                ## current model configuration
                model_initialization_config = self.models_initialization_config[model_serial_number]  ## current model configuration
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY], ##returns the lib and i th model
                                                            class_name=model_initialization_config[CLASS_KEY]    ## lib is importted and returned                
                                                            ) ## returns eg : LogisticRegression from sklearn.linear_model
                # model_obj_ref is  LogisticRegression now
                 
                model = model_obj_ref() ## giving parenthesis to make syntax it a function model
                print(model)
                
                # logistic_regression_model = LogisticRegression()
                # Create an instance of the LogisticRegression class
                

                # Now you can use the logistic_regression_model for further operations
                # For example, you can fit it to your data
                # logistic_regression_model.fit(X_train, y_train)


                                                     ## setting value of parameter fit_intercept to True
                if PARAM_KEY in model_initialization_config:  ## now we move on to update 3rd paramnter of the model config
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,  ## now model is updated with gicen params: fit_interpct =True
                                                                  property_data=model_obj_property_data)
                    print(model)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}" ## to string
                print(model_name)
                # eg: sklearn.linear_model.LogisticRegression
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )             ## each model updated with imported lib, updated with params
                                                                                   ## will now be stored in list
                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,   ## grid search opertion for single model recieved as parammter
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:      ## for every model in list , grid serach action is perfomed as below
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            print(e)
            raise CreditException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score         ## based on best score/base accuracy we are selecting model

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_best_model(self, X, y,base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()            
            logging.info(f"Initialized model: {initialized_model_list}")

            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            ## now passing the grid search model list which contains deatils like r2score, best params, etc of each models as list inorder 
            ## to select best one- shown below
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise CreditException(e, sys)