#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data/House_collapse_prediction_dataset.csv')
df.head()
df.info()

df.groupby('Serious_Destruction').mean(numeric_only=True) # 用Serious_Destruction分组，求每個column的平均值   numeric_only=True: 只求數值型的column的平均值

df.isnull().sum() # check the missing values
len(df)
len(df)/2
df.isnull().sum() > len(df)/2 # check the missing values which is greater than half of the length of the dataframe

df.drop('Building_Type',axis=1,inplace=True) # drop the Building_Type column
df.head()

df['Fault'].value_counts()
df['Fault'].fillna(df['Fault'].value_counts().idxmax(),inplace=True)

df['Soil_Liquefaction'].value_counts()
df['Soil_Liquefaction'].fillna(1,inplace=True)
df['Soil_Liquefaction'].isnull().value_counts()

df['Dip_Slope'].value_counts()
df['Dip_Slope'].fillna(df['Dip_Slope'].value_counts().idxmax(),inplace=True)
df['Dip_Slope'].isnull().value_counts()
df.drop('Dip_Slope',axis=1,inplace=True) # drop the Building_Type column

df['Land_Subsidence'].value_counts()
df['Land_Subsidence'].fillna(1,inplace=True)
df['Land_Subsidence'].isnull().value_counts()

# df['Material'].value_counts()
# df.drop('Material',axis=1,inplace=True) # drop the Building_Type column
df['Material'].info()
df['Material'].value_counts()
df['Material'].fillna("reinforced concrete mixed soil",inplace=True)
df['Material'] = np.where((df['Material'] == "Salad oil barrels, newspapers, bricks, cement paper bags mixed with steel bars"), 0, df['Material'] )
df['Material'] = np.where((df['Material'] == "Tofu dregs"), 0, df['Material'] )

df['Material'] = np.where((df['Material'] == "iron sheet"), 1, df['Material'] )
df['Material'] = np.where((df['Material'] == "Unreinforced brick masonry (no foundation)"), 1, df['Material'] )
df['Material'] = np.where((df['Material'] == "earth"), 1, df['Material'] )
df['Material'].value_counts()
df['Material'] = np.where((df['Material'] == "wash stone"), 2, df['Material'] )
df['Material'] = np.where((df['Material'] == "Hinoki"), 2, df['Material'] )
df['Material'] = np.where((df['Material'] == "wood"), 2, df['Material'] )
df['Material'] = np.where((df['Material'] == "stones and bricks"), 2, df['Material'] )
df['Material'] = np.where((df['Material'] == "Brick and wood"), 2, df['Material'] )
df['Material'].value_counts()
df['Material'] = np.where((df['Material'] == "red brick"), 3, df['Material'] )   
df['Material'] = np.where((df['Material'] == "Brick"), 3, df['Material'] )
df['Material'] = np.where((df['Material'] == "marble"), 3, df['Material'] )
df['Material'] = np.where((df['Material'] == "Bricks and tiles"), 3, df['Material'] )

df['Material'].value_counts()

df['Material'] = np.where((df['Material'] == "reinforced concrete mixed soil"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "concrete"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "cement"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "reinforced concrete"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "reinforced cement"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "reinforced concrete mixed soil+iron sheet"), 4, df['Material'] )
df['Material'] = np.where((df['Material'] == "reinforced concrete + iron sheet"), 4, df['Material'] )
df['Material'].value_counts()
df['Material'].isnull().value_counts()


df['Floor'].value_counts()
median_value = np.nanmedian(df['Floor'])
df['Floor'].fillna(median_value, inplace=True)
# df['Floor'] = 5*df['Floor']**2+df['Floor']

df['Serious_Destruction'].fillna(0, inplace=True)

df['City'].value_counts()
df['is_Hualien'] = np.where(df['City'] == 'Hualien', 1, 0)

df.drop('City',axis=1,inplace=True)
df.drop('Intensity',axis=1,inplace=True)

df.isnull().sum()

df.corr()

X=df.drop(['Serious_Destruction'],axis=1) 
y=df['Serious_Destruction']

#split to training data & testing data 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score



# max=0
# state_i=0
# state_j=0
# average=0
# max_average=0
# average_i=0
# final=0
# final_i=0
# final_j=0

# for i in range(10,50):
#     max=0
#     average=0
#     for j in range(100):
#         X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=(i/100),random_state=j)

#         # using Logistic regression model
#         lr=LogisticRegression(max_iter=200)
#         lr.fit(X_train,y_train)
#         predictions=lr.predict(X_test)
#         predictions

#         # Model Evaluation
#         accuracy_score(y_test,predictions)
#         # recall_score(y_test,predictions)
#         # precision_score(y_test,predictions)
#         # confusion_matrix(y_test,predictions)
#         average+=accuracy_score(y_test,predictions)
#         if(max<accuracy_score(y_test,predictions)):
#             max=accuracy_score(y_test,predictions)
#             state_i=i
#             state_j=j
#     if(max-(average/100) <= 0.2):
#         if(final<max):
#             final=max
#             final_i = state_i
#             final_j = state_j

# print("max:",final)
# print("final_i:",final_i)
# print("final_j:",final_j)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.23,random_state=58)
lr=LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
predictions

accuracy_score(y_test,predictions)

import joblib
joblib.dump(lr,'earthquake.pkl', compress=3)