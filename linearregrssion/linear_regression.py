import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from LinearRegression import DataSet


df = DataSet('../train.csv')
st.title('House Pricing Regression Model')
st.image('../house-prices.jpg')
st.dataframe(df.df)
feature_selected = st.selectbox('Select feature',df.df.select_dtypes(include=['int64','float64']).columns)

build_btn = st.button('Build Model With this Feature')
if build_btn:
    df.build_model(feature_selected,'SalePrice')
    fig = df.plotScatter(feature_selected,'SalePrice')
    metrics_df = df.calculate_metrics(feature_selected,'SalePrice')
    st.dataframe(metrics_df)
    st.pyplot(fig)