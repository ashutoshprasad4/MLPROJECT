import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_obj,evaluate_model




@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Initiating Model Trainer")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model={
                "LinearRegression":LinearRegression(),
                "DecisionTree":DecisionTreeRegressor(),
                "RandomForest":RandomForestRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "XGBoost":XGBRegressor(),
                "KNN":KNeighborsRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
            }
            params={
                "LinearRegression": {},
                "DecisionTree": {
                    
                    
                    "max_depth": [None, 10, 20, 30, 40, 50],
                    
                    
                },
                "RandomForest": {
                    "n_estimators": [100, 200, 300, 400, 500],
                   
                    "max_depth": [None, 10, 20, 30, 40, 50],
                    
                   
                },
                "GradientBoosting": {
                    
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    
                    
                    
                    "max_depth": [3, 4, 5, 6, 7, 8],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    
                },
                "XGBoost": {
                    
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    "max_depth": [3, 4, 5, 6, 7, 8],
                   
                    
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    
                    
                },
                "CatBoost": {
                    
                    "depth": [3, 4, 5, 6, 7, 8],
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                    
                },
            }

            model_report:dict=evaluate_model(models=model,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,params=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=model[best_model_name]
            if best_model_score<0.6:
                raise custom_exception("No model is performing well")
            logging.info(f"best found model on both test and train dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(Y_test,predicted)
            return r2_square
        except Exception as e:
            
            raise custom_exception(e,sys)
        
