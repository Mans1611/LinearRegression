from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score


class DataSet():
    model = LinearRegression()
    def __init__(self,path:str):
        self.df = pd.read_csv(path,index_col='Id') ## setting the index to be the Id
        
    def column_details(self,column_name):
        print(self.df[column_name].unique())
        print(self.df[column_name].info()) 
        
    def build_model(self,feature,target):
        self.model.fit(self.df[[feature]],self.df[[target]])
        score = self.model.score(self.df[[feature]],self.df[[target]])
        print(round(score*100,2))
        return self.model
    
    def plotScatter(self,feature,target):
        fig = plt.figure(figsize=(10,8))
        sns.scatterplot(x=self.df[feature],y=self.df[target])
        y=self.model.coef_[0][0] * self.df[feature] + self.model.intercept_
        sns.lineplot(x=self.df[feature],y=y ,color='r',linestyle='--');
        return fig
    
    def plot_shape_line(self,feature,target):
        agg_year_built = self.df[[feature,target]].groupby(feature).agg(['mean'])
        sns.lineplot(x=agg_year_built.index,y=agg_year_built[target]['mean'])
    
    def build_model(self,X,y):
        cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
        num_pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            MinMaxScaler())
        cat_attr = list(X.select_dtypes(include=[object]).columns)
        nums_attr = list(X.select_dtypes(include=[np.number]).columns)
          
        column_transformer = ColumnTransformer(  ## for making a parallel pipeline, one for categorical and one for numerical data.
            [('num',num_pipeline,nums_attr),
            ('cat',cat_pipeline,cat_attr)]
        )
        self.model = make_pipeline(column_transformer,LinearRegression())
        self.model.fit(X,y)
        return self.model.decision_function
    
    def calculate_metrics(self,feature,target):
        y_pred = self.model.predict(self.df[[feature]])
        return pd.DataFrame({
            'r2_score' :  [f'{round((r2_score(self.df[[target]],y_pred) * 100),2)}%'],
            'Mean Squared Error' :  [round(mean_squared_error(self.df[[target]],y_pred),2)],
            'Mean Absolute Error' : [round(mean_absolute_error(self.df[[target]],y_pred),2)],
        })