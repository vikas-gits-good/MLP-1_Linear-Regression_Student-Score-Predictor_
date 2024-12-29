import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Attempting to read data")
            self.df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Successfully read data as DataFrame")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            logging.info("Initiating Train-Test split")
            train_set, test_set = train_test_split(
                self.df, test_size=0.2, random_state=47
            )
            logging.info("Finished Train-Test split")

            self.df.to_csv(
                path_or_buf=self.ingestion_config.raw_data_path,
                index=False,
                header=True,
            )
            logging.info("Finished saving raw data")

            train_set.to_csv(
                path_or_buf=self.ingestion_config.train_data_path,
                index=False,
                header=True,
            )
            logging.info("Finished saving train data")

            test_set.to_csv(
                path_or_buf=self.ingestion_config.test_data_path,
                index=False,
                header=True,
            )
            logging.info("Finished saving test data")
            logging.info("Finished data ingestion")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.info(f"Error in data_ingestion.py: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Data Ingestion
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    # Data Transformation
    data_transf = DataTransformation()
    train_set, test_set, _ = data_transf.initiate_data_transformation(
        train_path=train_path, test_path=test_path
    )

    # Model Trainer
    mdl_trn = ModelTrainer()
    r2_score, model_name = mdl_trn.initiate_model_trainer(
        ary_train=train_set, ary_test=test_set
    )
    print(f"Best model is '{model_name}' with r2_score of {r2_score*100:.2f}%")
