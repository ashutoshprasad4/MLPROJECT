import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preproccesor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
            self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['writing score','reading score']
            categorical_columns=[
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                         
            ]
                    )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
            ]
            )
            logging.info("Data Transformation object created successfully")
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        

            
        except Exception as e:
             raise custom_exception(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data successfully")
            logging.info("Obtaining preprocessor object")

            preproccesor_obj=self.get_data_transformer_object()
            target_column='math score'
            numerical_columns=['writing score','reading score']

            input_features_train_df=train_df.drop(target_column,axis=1)
            target_column_train_df=train_df[target_column]

            input_features_test_df=test_df.drop(target_column,axis=1)
            target_column_test_df=test_df[target_column]
            logging.info("Fitting preprocessor object on train data")
            preproccesor_obj.fit_transform(input_features_train_df)
            input_features_train_df_transformed=preproccesor_obj.fit_transform(input_features_train_df)
            input_features_test_df_transformed=preproccesor_obj.transform(input_features_test_df)
            train_arr=np.c_[
                input_features_train_df_transformed,
                np.array(target_column_train_df)
            ]
            test_arr=np.c_[
                input_features_test_df_transformed,
                np.array(target_column_test_df)
            ]
            logging.info("Saving preprocessor object")

            save_obj(
                 file_path=self.data_transformation_config.preproccesor_obj_file_path,
                 obj=preproccesor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preproccesor_obj_file_path
            )

        except Exception as e:
            raise custom_exception(e,sys)
           

        
    

    
 