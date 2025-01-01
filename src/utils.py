import os
import sys
import dill

# import numpy as np
# import pandas as pd
from typing import Tuple

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path: str, obj):
    """Method to save the preprocessed data to file

    Args:
        file_path (str): File path directory where the data will be stored
        obj (array): Test & Training array sets

    Raises:
        CustomException: Error while saving data
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f"Error occured in utils.py: {e}")
        raise CustomException(e, sys)


def evaluate_model(
    X_train, Y_train, X_test, Y_test, Models, param
) -> Tuple[dict, dict]:
    """Method that evaluates models and returns a dictionary of r2_scores

    Args:
        X_train (array): Independant Variable training dataset
        Y_train (array): Dependant Variable training dataset
        X_test  (array): Independant Variable test dataset
        Y_test  (array): Dependant Variable test dataset
        Models  (dict) : Model name and Model Function
            Models = {
            'Model-1 Name' : Model_1_Function(),
            'Model-2 Name' : Model_2_Function(),
            'Model-3 Name' : Model_3_Function(),
            }
        param dict(dict): Dict of parameters for each of the models

    Returns:
        (Model-Score_dict, Best-Model-Score-dict)
        Model_Score (dict): Dictionary of Model name and r2_scores
        Best_Model_Score (dict): Dictionary with Model name and list of r2_scores of pred and test

    Raises:
        CustomException: Error while evaluating models
    """
    try:
        Model_scores = {}
        for i in range(len(list(Models))):
            model_name = list(Models.keys())[i]
            logging.info(f'Started training "{model_name}" model')

            model = list(Models.values())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate best model based on r2_score on test set
            model_train_score = r2_score(y_true=Y_train, y_pred=y_train_pred)
            model_test_score = r2_score(y_true=Y_test, y_pred=y_test_pred)

            Model_scores[model_name] = [model_train_score, model_test_score]
            logging.info(
                f'Finished training "{model_name}" model with r2_score {model_test_score*100:.2f}%'
            )
        # Rearrange in descending order and extract the max score model as dict
        Model_scores = dict(
            sorted(Model_scores.items(), key=lambda item: item[1][1], reverse=True)
        )

        Max_model = dict(
            sorted(Model_scores.items(), key=lambda item: item[1][1], reverse=True)[:1]
        )

        logging.info(
            f'Best model is "{list(Max_model.keys())[0]}" with r2_score {list(Max_model.values())[0][1]*100:.2f}%'
        )

        return (Model_scores, Max_model)

    except Exception as e:
        logging.info(f"Error in utils.py: {e}")
        raise CustomException(e, sys)
