import os 
import sys 

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from src.utils import saved_files
from src.exception import CustomException
from src.logger import logging 
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self , train_arr , test_arr):
        try:
            logging.info("Extracting the train and test")
            x_train , y_train , x_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'LogisticRegression' : LogisticRegression(),
                'SVC': SVC(),
                'KNeighborsClassifier' : KNeighborsClassifier(),
                'AdaBoostClassifier' : AdaBoostClassifier(),
                'GradientBoostingClassifier' : GradientBoostingClassifier(),
                'XGBClassifier' : XGBClassifier()
            }
            
            params = {
                'DecisionTreeClassifier': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'RandomForestClassifier': {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga'],
                    'max_iter': [1000]
                },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto']
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },
                'XGBClassifier': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
            }

            model_report = evaluate_model(x_train=x_train , x_test=x_test , y_train=y_train , y_test=y_test , models=models , params=params)

            best_model_name = model_report.idxmax()
            best_model_score = model_report.max()

            best_model = models[best_model_name]

            print("Model evaluation report:", model_report)
            if best_model_score < 0.4:
                print(f"Warning: Best model score {best_model_score} is less than 0.4")
            logging.info("model is performaing good")

            saved_files(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )
            prediction = best_model.predict(x_test)

            accuracy = accuracy_score(y_test , prediction)
            return accuracy

            logging.info("Model training completed")

        except Exception as e:
            raise CustomException(e,sys)