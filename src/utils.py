import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj):
    """Method to save the preprocessed data to file

    Args:
        file_path (str): File path directory where the data will be stored
        obj (array(float)): Test & Training array sets

    Raises:
        CustomException: Error while saving data
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error occured in utils.py: {e}")
        raise CustomException(e, sys)
