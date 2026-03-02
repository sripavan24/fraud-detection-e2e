import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from src.utils import save_object 


@dataclass
class ModeltrainConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")
  
  
class modeltrain:
    def __init__ (self):
        self.train_model_file_config = ModeltrainConfig()        
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Start model building")

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            scale_pos_weight = 227459 / 387

            xgb = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )

            xgb.fit(x_train, y_train)

            xgb_pred = xgb.predict(x_test)

            score = accuracy_score(y_test, xgb_pred)

            if score < 0.6:
                raise CustomException("Model performance is too low", sys)

            save_object(
                file_path=self.train_model_file_config.train_model_file_path,
                obj=xgb
            )

            return score

        except Exception as e:
            raise CustomException(e, sys)
    