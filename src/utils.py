import os
import sys

import pandas as pd
import numpy as np
from src.exception import custom_exception
import dill
from sklearn.metrics import r2_score



def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise custom_exception(e,sys)



def evaluate_model(models,X_train,Y_train,X_test,Y_test):
    try:
        model_report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,Y_train)
            Y_train_pred=model.predict(X_train)
            Y_test_pred=model.predict(X_test)
            train_model_score=r2_score(Y_train,Y_train_pred)
            test_model_score=r2_score(Y_test,Y_test_pred)
            model_report[list(models.keys())[i]]=test_model_score

        return model_report             

        
    except Exception as e:
        raise custom_exception(e,sys)

