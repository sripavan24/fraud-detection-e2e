import sys
import pandas as pd
from src.exception import CustomException

from src.utils import load_object

import pickle


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform input data
            data_scaled = preprocessor.transform(features)

            # Predict
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Time,
                 V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                 V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
                 V21, V22, V23, V24, V25, V26, V27, V28,
                 Amount):

        self.Time = Time
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Time": [self.Time],
                "V1": [self.V1],
                "V2": [self.V2],
                "V3": [self.V3],
                "V4": [self.V4],
                "V5": [self.V5],
                "V6": [self.V6],
                "V7": [self.V7],
                "V8": [self.V8],
                "V9": [self.V9],
                "V10": [self.V10],
                "V11": [self.V11],
                "V12": [self.V12],
                "V13": [self.V13],
                "V14": [self.V14],
                "V15": [self.V15],
                "V16": [self.V16],
                "V17": [self.V17],
                "V18": [self.V18],
                "V19": [self.V19],
                "V20": [self.V20],
                "V21": [self.V21],
                "V22": [self.V22],
                "V23": [self.V23],
                "V24": [self.V24],
                "V25": [self.V25],
                "V26": [self.V26],
                "V27": [self.V27],
                "V28": [self.V28],
                "Amount": [self.Amount],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise Exception(e)