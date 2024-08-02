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
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold

class DataSet():
    model = LinearRegression()
    def __init__(self,path:str):
        self.df = pd.read_csv(path,index_col='Id') ## setting the index to be the Id
        
    def wrangle_data(self,df,columns_trim=[],trim_end=0.95):
        try:
            drop_columns = set(['Alley','PoolQC','Fence','SaleType','OpenPorchSF','GarageYrBlt','YrSold','PoolArea','Street','LandContour','Utilities','LandSlope','Condition1','Condition2','SaleCondition','EnclosedPorch','3SsnPorch','SaleType','MoSold','HalfBath','RoofMatl','GarageCars','WoodDeckSF','EnclosedPorch','Functional','FireplaceQu','PavedDrive','MSZoning','LotConfig','LotFrontage','LotArea','OverallCond','MasVnrType','MiscFeature','ScreenPorch','MiscVal','BsmtHalfBath','LowQualFinSF','BsmtUnfSF','BsmtExposure','Fireplaces','Exterior2nd','GarageCond','KitchenAbvGr','BedroomAbvGr','BsmtFullBath','BsmtFinType2','BsmtFinSF2','ExterCond','BsmtFinSF1','BsmtFinType1','BsmtCond','YearRemodAdd','MasVnrArea','GarageFinish','GarageQual',
                'LotShape','Exterior1st','MSSubClass','TotalBsmtSF'])
            df = df.drop(columns=drop_columns)
            df = df[ df['SalePrice'] < df['SalePrice'].quantile(0.95)]
            df = df.dropna()
            
            for col_tri in columns_trim: ## removing the outliers of the features data
                df = df[ df[col_tri] < df[col_tri].quantile(trim_end)]
            #df['allarea'] = df.apply(lambda row: row['1stFlrSF'] + row['2ndFlrSF'],axis=1) ## combining the two features togther
            df.drop(columns=['1stFlrSF','2ndFlrSF'],inplace=True)
            return df
        except (KeyError, AttributeError) as e  :
            return self.wrangle_data(df.drop(columns=drop_columns))

        
    def column_details(self,column_name):
        print(self.df[column_name].unique())
        print(self.df[column_name].info()) 
        
    def build_model(self,feature,target):
        '''
            this building model for building unifeature which is selected in 
            streamlit.
        '''
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
        
    def split_data(self):
        X = self.df.drop(columns=['SalePrice'])
        y = self.df['SalePrice']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test
        
    def build_preprocess_pipeline(self):
        ordinal_attr = set(['BsmtQual','KitchenQual','ExterQual','BsmtQual','HeatingQC'])
        cat_attr = set(list(self.X.select_dtypes(include=[object]).columns))
        cat_attr = list(set([item for item in cat_attr if item not in ordinal_attr ]))
        nums_attr = list(set(list(self.X.select_dtypes(include=[np.number]).columns)))
        ordinal_attr = list(ordinal_attr)
        
        cat_pipeline = OneHotEncoder(handle_unknown='infrequent_if_exist')
        ordinal_pipe = make_pipeline(OrdinalEncoder(),StandardScaler())
        num_pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            MinMaxScaler())
        return ColumnTransformer([  ## for making a parallel pipeline, one for categorical and one for numerical data.
            ('num',num_pipeline,nums_attr),
            ('cat',cat_pipeline,cat_attr),
            ('ordinal',ordinal_pipe,ordinal_attr)
            ])
    
    
    def build_model_feature_lr(self):
        column_transformer = self.build_preprocess_pipeline()
        self.model = make_pipeline(column_transformer,LinearRegression())
        X_train,X_test,y_train,y_test = self.split_data()
        self.model.fit(X_train,y_train)
        return r2_score(y_test,self.model.predict(X_test))
    
    def calculate_metrics(self,feature,target):
        y_pred = self.model.predict(self.df[[feature]])
        return pd.DataFrame({
            'r2_score' :  [f'{round((r2_score(self.df[[target]],y_pred) * 100),2)}%'],
            'Mean Squared Error' :  [round(mean_squared_error(self.df[[target]],y_pred),2)],
            'Mean Absolute Error' : [round(mean_absolute_error(self.df[[target]],y_pred),2)],
        })
        
    def make_cross_validation(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(self.model,self.X,self.y,cv=kf)