from cgi import test
from sklearn import preprocessing
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity.config_entity import DataTransformationConfig 
from insurance.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from insurance.constant import *
from insurance.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data





class FeatureGenerator(BaseEstimator, TransformerMixin):              ## inheriting classes

    def __init__(self,age_ix=3,children_ix=5,columns=None):

        try:            
            self.columns = columns
            if self.columns is not None:                                               
                age_ix = self.columns.index(COLUMN_AGE)
                children_ix = self.columns.index(COLUMN_CHILDREN)
            else:
                self.age_ix = age_ix
                age_ix= self.age_ix
                self.children_ix = children_ix
                children_ix=self.children_ix

            self.age_ix = age_ix
            self.children_ix = children_ix  
          

        except Exception as e:
            raise InsuranceException(e, sys) from e
 


    def fit(self, X, y=None):

        #return X
        return self

    def transform(self, X, y=None):
        try:
            children_age_ratio = X[:, self.children_ix] / \
                                 X[:, self.age_ix]   
            # #                                              ##dividing one entire column with other
            # # ##population_per_household = X[:, self.population_ix] / \
            # #  ##                          X[:, self.households_ix]
            # # ##if self.add_bedrooms_per_room:                                   ## if true as per above initialization
            # #     bedrooms_per_room = X[:, self.total_bedrooms_ix] / \
            # #                         X[:, self.total_rooms_ix]
            # #     generated_feature = np.c_[                              ### joiinng all columns with new 
            # #         X, bmi_age_ratio, population_per_household, bedrooms_per_room]
            # # else:
            generated_feature = np.c_[ 
                    X, children_age_ratio]

            # generated_feature = np.c_[ 
            #         X]

            return generated_feature
            ##return X
        except Exception as e:
            print(e)
            raise InsuranceException(e, sys) from e





class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise InsuranceException(e,sys) from e

    

    def get_data_transformer_object(self)->ColumnTransformer:  ## returns tha preprocessing pickle file s
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path      ##schema.yaml file

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            columns = dataset_schema[COLUMNS]
            columns_input = dataset_schema[COLUMNS_INPUT]


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                  ('feature_generator', FeatureGenerator(
                   columns= numerical_columns
                     )),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('one_hot_encoder', OneHotEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ('num_pipeline', num_pipeline, numerical_columns),
                
            ])
            return preprocessing

        except Exception as e:
            raise InsuranceException(e,sys) from e   


    def initiate_data_transformation(self)->DataTransformationArtifact: 
        try:
            logging.info(f"Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object() ## getting pickle obj function


            logging.info(f"Obtaining training and test file path.")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path           ## getting files paths
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            
            
            ## getting data frame from the .csv in ingestion folder for transformation

            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)    ## loading data using utility functions
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)


            schema = read_yaml_file(file_path=schema_file_path)  ## getting schema

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)    ## fit and trasfrom input train features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)       ## trasfrom input test features

            ##train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]    ##trasfromed train df 
            
            ##test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)] 
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]          ##transormded test df 
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")      ## changing fileextension
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")       ## changing fileextension

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)  
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr) ## calling a util fucntion to save numpy array
                                                                                         ## and saving both trasformed  df's to trasfomred dir
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr) 

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path ## creating trasfomred dir

            logging.info(f"Saving preprocessing object.")

            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj) ## saving pre-procesd obj to trasfomred dir

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            print(e)
            raise InsuranceException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")