import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    pre_proc_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function performs data transformation on input data

        Raises:
            CustomException: Error while performing data transformation

        Returns:
            pre_proc: numerical and categorical columnar transformed data
        """
        logging.info("Entered data transformation method")
        try:
            logging.info("Initiating data transformation")
            num_cols = ["writing_score", "reading_score"]
            cat_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            logging.info("Initialising pipelines")
            num_pl = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Finished numerical pipelines")

            cat_pl = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Finished categorical pipelines")

            pre_proc = ColumnTransformer(
                [
                    ("num_pipeline", num_pl, num_cols),
                    ("cat_pipeline", cat_pl, cat_cols),
                ]
            )

            return pre_proc

        except Exception as e:
            logging.info(f"Error in data_transformation.py: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Entered data transformation initialisation method")
        try:
            df_train = pd.read_csv(train_path)
            logging.info("Successfully read train dataset")

            df_test = pd.read_csv(test_path)
            logging.info("Successfully read test dataset")

            logging.info("Obtaining preprocessing object")
            pre_proc = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            df_input_feat_train = df_train.drop(columns=[target_column_name])
            df_target_feat_train = df_train[target_column_name]

            df_input_feat_test = df_test.drop(columns=[target_column_name])
            df_target_feat_test = df_test[target_column_name]

            logging.info("Applying preprocessing object on training and test datasets")
            ary_ip_ft_train = pre_proc.fit_transform(df_input_feat_train)
            ary_ip_ft_test = pre_proc.transform(df_input_feat_test)

            ary_train = np.c_[ary_ip_ft_train, np.array(df_target_feat_train)]
            ary_test = np.c_[ary_ip_ft_test, np.array(df_target_feat_test)]

            logging.info("Saving preprocessed object")
            save_object(
                file_path=self.data_transformation_config.pre_proc_obj_file_path,
                obj=pre_proc,
            )

            return (
                ary_train,
                ary_test,
                self.data_transformation_config.pre_proc_obj_file_path,
            )

        except Exception as e:
            logging.info(f"Error in data_transformation.py: {e}")
            raise CustomException(e, sys)
