from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model, save_obj

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "XGBRegressor": XGBRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR()
            }

            # Hyperparameter grids for tuning
            param_distributions = {
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                "XGBRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }

            best_models = {}
            for name, model in models.items():
                logging.info(f"Training and tuning {name}")
                if name in param_distributions:
                    # Apply RandomizedSearchCV for hyperparameter tuning
                    rs = RandomizedSearchCV(
                        model,
                        param_distributions=param_distributions[name],
                        n_iter=10,
                        scoring='r2',
                        cv=5,
                        verbose=2,
                        n_jobs=-1,
                        random_state=42
                    )
                    rs.fit(X_train, y_train)
                    best_models[name] = rs.best_estimator_
                    logging.info(f"{name} best params: {rs.best_params_}")
                else:
                    # Train models without hyperparameter tuning
                    model.fit(X_train, y_train)
                    best_models[name] = model

            # Evaluate tuned models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, best_models)
            print(model_report)

            # Select best model based on R2 score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = best_models[best_model_name]

            print(f"Best model found: {best_model_name}, R2 score: {best_model_score}")
            logging.info(f"Best model found: {best_model_name}, R2 score: {best_model_score}")

            # Save the best model
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
