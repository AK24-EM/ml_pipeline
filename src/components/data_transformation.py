from genericpath import exists
import sys
import pandas as pd
import os
import sys 
import dataclasses as dataclasses

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import saved_files
@dataclasses.dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join("artifacts" , "preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()

    def get_data_tranform_obj(self):
        try:
            numerical_cols = ["Year", "Attendance", "Date", "month", "Home_Team_Avg_Goals_Scored", "Away_Team_Avg_Goals_Scored", "Home_Team_Win_Rate", "Away_Team_Win_Rate"]

            categorical_cols = ["Stage", "Home Team Name", "Away Team Name"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy='median')),
                    ("scaler" , StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="most_frequent")),
                    ("onehotencode" , OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            logging.info("cateogrical encoding completed")

            preprocessing_obj = ColumnTransformer(
                [
                    ("numerical_pipeline" , num_pipeline , numerical_cols) ,
                    ("categorical_pipeline" , categorical_pipeline , categorical_cols)
                ]
            )

            return preprocessing_obj


        except Exception as e:
            raise CustomException(e , sys)

    def initiate_data_transformation(self , train_path , test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the traina and test data")

            preprocessing_obj = self.get_data_tranform_obj()

            target_column = 'Result'

            input_feature_train= train_df.drop(columns=[target_column])
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns=[target_column])
            target_feature_test = test_df[target_column]

            logging.info("applying preprocessing object on training and testing data")

            preprocessed_train = preprocessing_obj.fit_transform(input_feature_train)
            preprocessed_test = preprocessing_obj.transform(input_feature_test)

            le = LabelEncoder()
            target_feature_train = le.fit_transform(target_feature_train)
            target_feature_test = le.transform(target_feature_test)

            train_arc = np.c_[preprocessed_train , np.array(target_feature_train)]
            test_arc = np.c_[preprocessed_test , np.array(target_feature_test)]

            logging.info("Saved preprocessing saved")

            saved_files(
                file_path = self.data_transfromation_config.preprocessing_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arc , test_arc,
                self.data_transfromation_config.preprocessing_obj_file_path
            )

            
        except Exception as e:
            raise CustomException(e , sys)