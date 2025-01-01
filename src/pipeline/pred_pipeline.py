import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Attempting to predict score based on user inputs")

            # Get the model and the preprocessed data
            model_path = "artifacts/model.pkl"
            peprc_path = "artifacts/preprocessor.pkl"

            # Load the model and the preprocessed data
            model = load_object(file_path=model_path)
            peprc = load_object(file_path=peprc_path)

            # Transform the data and make prediction
            data_scaled = peprc.transform(features)
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            logging.info(f"Error in pre_pipeline.py: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race: str,
        parental_education: str,
        lunch: str,
        test_prep: str,
        read_score: int,
        write_score: int,
    ):
        logging.info("Initialising data in prediction pipeline")
        self.gender = gender
        self.race = race
        self.prnt_educ = parental_education
        self.lunch = lunch
        self.test_prep = test_prep
        self.read_sc = read_score
        self.writ_sc = write_score

    def get_data_as_frame(self):
        logging.info("Entered get data as DataFrame method")
        try:
            logging.info("Preparing DataFrame based on user input")
            cust_data_dict: dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race],
                "parental_level_of_education": [self.prnt_educ],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_prep],
                "reading_score": [self.read_sc],
                "writing_score": [self.writ_sc],
            }
            return pd.DataFrame(data=cust_data_dict)

        except Exception as e:
            logging.info(f"Error in pred_pipeline.py: {e}")
            raise CustomException(e, sys)
