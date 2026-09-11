import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
import dill
def save_object(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            # Predict Testing data
            y_test_pred=model.predict(x_test)

            # Get model score
            test_model_score=r2_score(y_test,y_test_pred)
            train_model_score=r2_score(y_train,y_train_pred)

            report[list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)