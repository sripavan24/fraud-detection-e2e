import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import sys
import os
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import save_object 
@dataclass

class DatatransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts","preprocessor.pkl")
  
  
class Datatransformation:
    def __init__ (self):
        self.data_transformation_config = DatatransformationConfig()        
        
    def get_data_transform(self):
        try:
            df=pd.read_csv("notebook\\data\\creditcard.csv")
            
            df1 = df.drop("Class", axis=1).columns            
            logging.info("num scaling is stated")
            
            Pipeline1=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaling",StandardScaler())
            ])
            
            
            preprosser=ColumnTransformer([
                ("num_pipline",Pipeline1,df1)
                ]
            )
            
            logging.info("over to conb")
            return preprosser
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path) 
            test_df=pd.read_csv(test_path) 
            
            logging.info("read train and test data completed")
            
            preprocessing_obj=self.get_data_transform()
            
            target_colunm_name="Class"
            input_feature_train_df=train_df.drop(columns=[target_colunm_name],axis=1)
            target_feature_train_df=train_df[target_colunm_name]
            
            
            input_feature_test_df=test_df.drop(columns=[target_colunm_name],axis=1)
            target_feature_test_df=test_df[target_colunm_name]
            logging.info("appling prepossing onj on training datafram and testing datafram")
            
            
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("saved preprossing object")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_path,
                        obj=preprocessing_obj)
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path)
            
        except EOFError as e:
            raise CustomException(e,sys)

                                          
    