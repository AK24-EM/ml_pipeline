import os 
import sys 

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_path_data = os.path.join("artifacts" , "train.csv")
    test_path_data = os.path.join("artifacts" , "test.csv")
    raw_path_data = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_data = DataIngestionConfig()

    def initiate_data_config(self):
        logging.info("entered the data ingestion initiated")
        try :
            df = pd.read_csv("/Users/aayushkamble/ml_pipeline/src/notebook/data/worldcup_claeaned2.csv")
            logging.info("Reading the data")

            os.makedirs(os.path.dirname(self.ingestion_data.train_path_data),exist_ok=True)
            df.to_csv(self.ingestion_data.raw_path_data , index=False , header=True)
            
            logging.info("Train Test split intiated")
            train_set, test_set = train_test_split(df,test_size=0.2 , random_state=42)

            train_set.to_csv(self.ingestion_data.train_path_data,index=True , header=True)

            test_set.to_csv(self.ingestion_data.test_path_data, index=True , header=True)

            logging.info("ingeston of the data is completed")
            
            return (
                self.ingestion_data.train_path_data,
                self.ingestion_data.test_path_data
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_config()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    from src.components.model_trainer import ModelTrainer
    model_trainer = ModelTrainer()
    accuracy = model_trainer.initate_model_training(train_arr, test_arr)
    print(f"Model accuracy: {accuracy}")