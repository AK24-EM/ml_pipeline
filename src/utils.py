import pandas as pd
import numpy as np

import os 
import sys

import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def saved_files(file_path , obj):
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        
        with open(file_path , 'wb') as file_obj:
            dill.dump(obj , file_obj)
    except Exception as e :
        raise CustomException(e , sys)

def evaluate_model(x_train, x_test, y_train, y_test, models , params):
    try:
        from sklearn.metrics import accuracy_score
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]

            grid = GridSearchCV(model , param , cv=5, n_jobs=-1)
            grid.fit(x_train , y_train)
            
            model.set_params(**grid.best_params_)
            model.fit(x_train,y_train)
            # model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        return pd.Series(report)
        
    except Exception as e:
        raise CustomException(e, sys)