import joblib
import numpy as np

house_model = joblib.load('collision.pkl')

def predict_collision(features):
    return house_model.predict(features)

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 186 entries, 0 to 185
# Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   縣市           186 non-null    int64  
#  1   斷層           186 non-null    float64
#  2   土壤液化有無(1/0)  186 non-null    float64
#  3   地層下陷有無(1/0)  186 non-null    float64
#  4   建材           186 non-null    int64  
#  5   樓層高度         186 non-null    float64
# dtypes: float64(4), int64(2)
# memory usage: 8.8 KB



s = input("Enter the features (縣市, 斷層, 土壤液化有無(1/0), 地層下陷有無(1/0), 建材, 樓層高度: ")
input_features = list(map(float, s.split(',')))

# Convert the input features to a numpy array
input_features = np.array(input_features).reshape(1, -1)
prediction = house_model.predict(input_features)
print(f"The prediction result is: {prediction[0]}")


from configparser import ConfigParser
import os

# Config Parser
config = ConfigParser()
config.read("config.ini")

#把密碼寫到環境變數
os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 直接問問題
result = llm.invoke("How to have a successful life?")
print(result.content)

from langchain_core.messages import HumanMessage, SystemMessage

# 輸入問題
user_input = input("Please enter your question: ")

model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
result = model.invoke(
    [
        #系統提示
        SystemMessage(content="若要使用預測系統，請輸入我要預測喔啾咪"),
        #加了input視窗
        HumanMessage(content=user_input),
    ]
)
print(result.content)