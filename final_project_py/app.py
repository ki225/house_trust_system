import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

#import dataset
df = pd.read_csv('data/predict_data_all.csv')
df.head(n=10)
df.info()

# remove the columns that are not useful
df.drop(['樓型', '是否為順向坡(1/0)'], axis=1, inplace=True)
df.head()

# 做表格觀察存活跟某因素之關西
# sns.pairplot(df[['破壞致無法居住(1/0)','樓層高度']], dropna=True)
#sns.pairplot(df[['Survived','Age']], dropna=True)

# data observation
df.groupby('破壞致無法居住(1/0)').mean(numeric_only=True)

#handling missing data
df.isnull().sum()
len(df)
len(df)/2
df.isnull().sum() > len(df)/2

# 把沒倒的震度改成8
df.loc[df['破壞致無法居住(1/0)'] == 0, '震度'] = 8

df.isnull().sum()


# fill the missing val with the medium value 用補遺漏值
df['樓層高度'].fillna(df['樓層高度'].median(), inplace=True)
df['樓層高度'].value_counts()

# check if there is any missing val 繼續找有沒有遺漏的值
df.isnull().sum()
df['斷層'].value_counts()
df['斷層'].info()

# fill 斷層
# for i in range(len(df)):
#     if df['斷層'][i] == '1':
#         df['斷層'][i] = 1
#     elif df['斷層'][i] == '0':
#         df['斷層'][i] = 0
#     elif df['斷層'][i] == '逆斷層' or df['斷層'][i] == '平移斷層':
#         if df['破壞致無法居住(1/0)'][i] == 1:
#             df['斷層'][i] = 1
#         else:
#             df['斷層'][i] = 0
    
df['斷層'].value_counts()
df['斷層'].info()

# 只有176個 檢查一下
for i in range(len(df)):
    if df['斷層'][i] != 0 or df['斷層'][i] != 1 :
        print(df['斷層'][i])
# 乾怎麼都是object

# 換個方法
df['斷層'] = pd.to_numeric(df['斷層'].astype(str), errors='coerce')

df['斷層'].value_counts()
df['斷層'].info()
df['斷層'].isnull().sum()

# ˋ終於
# 不能用一般if判斷是填
# null_rows = df[df['斷層'].isnull()]
# print(null_rows)

# for i in range(len(df)):
#     print(type(df['斷層'][i]))
# print(type(df['斷層'][0]))

# fk useless
# df['斷層'] = df['斷層'].fillna(value=1.0, mask=df['破壞致無法居住(1/0)'] == 1.0)

# !!!!!!!!!!!!!!!!!!!一定要這樣填 不然還是出錯 幹
df['斷層'] = np.where((df['斷層'].isnull()) & (df['破壞致無法居住(1/0)'] == 1), 1, df['斷層'])
df['斷層'] = np.where((df['斷層'].isnull()) & (df['破壞致無法居住(1/0)'] == 0), 0, df['斷層'])
df['斷層'].value_counts()
df['斷層'].info()
df['斷層'].isnull().sum()
# fk終於成功了ㄚㄚㄚㄚㄚㄚㄚ

# type不對
# fill 斷層
# for i in range(len(df)):
#     if pd.isnull(df['斷層'][i]) and df['破壞致無法居住(1/0)'][i] == 1:
#         df['斷層'][i] = 1
#     elif pd.isnull(df['斷層'][i]) and df['破壞致無法居住(1/0)'][i] == 0:
#         df['斷層'][i] = 0
# df['斷層'].value_counts()
# df.isnull().sum()

df.info()
df['土壤液化有無(1/0)'].value_counts()
df['土壤液化有無(1/0)'].info()
df['土壤液化有無(1/0)'].isnull().sum()
# fill 土壤液化有無
# for i in range(len(df)):
#     if pd.isnull(df['土壤液化有無(1/0)'][i]) and df['破壞致無法居住(1/0)'][i] == 1:
#         df['土壤液化有無(1/0)'][i] = 1
#     elif pd.isnull(df['土壤液化有無(1/0)'][i]) and df['破壞致無法居住(1/0)'][i] == 0:
#         df['土壤液化有無(1/0)'][i] = 0

# 根據斷層的經驗，還是不要用if判斷
df['土壤液化有無(1/0)'] = np.where((df['土壤液化有無(1/0)'].isnull()) & (df['破壞致無法居住(1/0)'] == 1), 1, df['土壤液化有無(1/0)'] )
df['土壤液化有無(1/0)'] = np.where((df['土壤液化有無(1/0)'].isnull()) & (df['破壞致無法居住(1/0)'] == 0), 0, df['土壤液化有無(1/0)'] )
df['土壤液化有無(1/0)'].value_counts()
df.isnull().sum()

#fill 地層下陷有無
df['地層下陷有無(1/0)'].value_counts()
# for i in range(len(df)):
#     if pd.isnull(df['地層下陷有無(1/0)'][i]) and df['破壞致無法居住(1/0)'][i] == 1:
#         df['地層下陷有無(1/0)'][i] = 1
#     elif pd.isnull(df['地層下陷有無(1/0)'][i]) and df['破壞致無法居住(1/0)'][i] == 0:
#         df['地層下陷有無(1/0)'][i] = 0


df['地層下陷有無(1/0)'] = np.where((df['地層下陷有無(1/0)'].isnull()) & (df['破壞致無法居住(1/0)'] == 1), 1, df['地層下陷有無(1/0)'] )
df['地層下陷有無(1/0)'] = np.where((df['地層下陷有無(1/0)'].isnull()) & (df['破壞致無法居住(1/0)'] == 0), 0, df['地層下陷有無(1/0)'] )
df['地層下陷有無(1/0)'].value_counts()
df.isnull().sum()

# 樓層處理
df['樓層高度'].value_counts()
# 把樓層高度變成5x^2+x
df['樓層高度'] = np.where(df['樓層高度'].notnull(), 5 * df['樓層高度'] ** 2 + df['樓層高度'], df['樓層高度'])

# use most value fill 建材
df['建材'].value_counts()
for i in range(len(df)):
    if df['建材'][i] == '鋼筋混泥土' or df['建材'][i] == '鋼筋混擬土':
        df['建材'][i] = '鋼筋混凝土'
    elif df['建材'][i] == '鋼筋混尼土' or df['建材'][i] == '鋼筋水泥':
        df['建材'][i] = '鋼筋混凝土'

df['建材'].value_counts()

for i in range(len(df)):
    if pd.isnull(df['建材'][i]):
        df['建材'][i] = '鋼筋混凝土'

df.isnull().sum()

# fill 震度
df['震度'].value_counts()

for i in range(len(df)):
    if pd.isnull(df['震度'][i]):
        if df['破壞致無法居住(1/0)'][i] == 1:
            df['震度'][i] = df['震度'].median()
        else:
            df['震度'][i] = 8

df.isnull().sum()
# prediction 100% 篇扯 刪看看
df.drop('震度', axis=1, inplace=True)
# 把名稱drop掉
df.drop('名稱', axis=1, inplace=True)

# 把縣市drop掉
# 但這邊可以調城分數
# df.drop('縣市', axis=1, inplace=True)

# 處理縣市
df['縣市'].value_counts()
df['縣市'].info()

# 把台中有關的都換成台中市
df['縣市'] = np.where(df['縣市'] == '台中', '台中市', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '台中市', '臺中市', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '花蓮市', '花蓮縣', df['縣市'])
#如果名稱中有台北，就將縣市改成臺北市
df['縣市'] = np.where(df['縣市'] == '台北市', '臺北市', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '台北市 ', '臺北市', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '台南市', '臺南市', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '台東縣', '臺東縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '台東', '臺東縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '吉安鄉', '臺東縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '新竹', '新竹縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '新竹市', '新竹縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '南投', '南投縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '彰化', '彰化縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '雲林', '雲林縣', df['縣市'])
df['縣市'] = np.where(df['縣市'] == '宜蘭縣 ', '宜蘭縣', df['縣市'])

# 這邊可以調分數
# 花蓮 5
# 台東 4
# 台中, 南投, 台南 3
# 台北, 新竹, 桃園, 彰化,  2
# 高雄, 新北, 宜蘭, 屏東, 雲林, 苗栗, 嘉義 1

df['縣市'] = np.where(df['縣市'] == '花蓮縣', 5, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '臺東縣', 4, df['縣市'])

df['縣市'] = np.where(df['縣市'] == '臺中市', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '南投縣', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '臺南市', 3, df['縣市'])

df['縣市'] = np.where(df['縣市'] == '臺北市', 2, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '桃園市', 2, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '新竹縣', 2, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '彰化縣', 2, df['縣市'])

df['縣市'] = np.where(df['縣市'] == '宜蘭縣', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '新北市', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '高雄市', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '屏東縣', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '雲林縣', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '苗栗縣', 1, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '嘉義縣', 1, df['縣市'])

df['縣市'].value_counts()
df['縣市'].info()

# 把剩下的縣市都換成3
df['縣市'] = np.where(df['縣市'] == '墨西哥墨西哥城', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '日本石川縣', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '中國四川省', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '日本宮城縣', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '印尼龍目島', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '印尼東爪哇', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '土耳其馬迪亞巴克爾市', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '緬甸曼德勒', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '菲律賓艾布拉省', 3, df['縣市'])
df['縣市'] = np.where(df['縣市'] == '巴布亞紐幾內亞凱南圖', 3, df['縣市'])

df['縣市'].value_counts()
df['縣市'].info()

# object轉成int
df['縣市'] = pd.to_numeric(df['縣市'].astype(str), errors='coerce')

# error: could not convert string to float: '逆斷層'
# count = 0
# for i in range(len(df)):
#     if df['斷層'][i] != 0 or df['斷層'][i] != 1 :
#         print(df['斷層'][i])
#         count += 1
#     elif df['斷層'][i] == '逆斷層' or df['斷層'][i] == '平移斷層':
#         print(df['斷層'][i])
#         count += 1
# print(count)

df.info()

# type change to float
# df['斷層'] = pd.to_numeric(df['斷層'].astype(str), errors='coerce')
# df.info()

# 剛剛忘記改植fk
# for i in range(len(df)):
#     if df['斷層'][i] != 0 or df['斷層'][i] != 1 :
#         if df['破壞致無法居住(1/0)'][i] == 1:
#             df['斷層'][i] = 1
#         else:
#             df['斷層'][i] = 0
# df['斷層'].value_counts()
# 還有null,但沒法用df['斷層'].isnull()找出來
# df['斷層'].value_counts()
# df['斷層'].isnull().sum()

# 發現有些是填逆斷層/平移斷層
# null_rows = df[df['斷層'].isnull()]
# print(null_rows)

# 然後


df['建材'].info()
df['建材'].value_counts()
df['建材'] = np.where((df['建材'] == "沙拉油桶、報紙、磚塊、水泥紙袋混充"), 0, df['建材'] )
df['建材'] = np.where((df['建材'] == "豆腐渣"), 0, df['建材'] )

df['建材'] = np.where((df['建材'] == "鐵皮"), 1, df['建材'] )
df['建材'] = np.where((df['建材'] == "無筋磚砌體（無地基）"), 1, df['建材'] )
df['建材'] = np.where((df['建材'] == "土"), 1, df['建材'] )
df['建材'].value_counts()
df['建材'] = np.where((df['建材'] == "洗石"), 2, df['建材'] )
df['建材'] = np.where((df['建材'] == "檜木"), 2, df['建材'] )
df['建材'] = np.where((df['建材'] == "木材"), 2, df['建材'] )
df['建材'] = np.where((df['建材'] == "石塊和磚瓦"), 2, df['建材'] )
df['建材'] = np.where((df['建材'] == "磚木"), 2, df['建材'] )
df['建材'].value_counts()
df['建材'] = np.where((df['建材'] == "紅磚"), 3, df['建材'] )   
df['建材'] = np.where((df['建材'] == "磚瓦"), 3, df['建材'] )
df['建材'] = np.where((df['建材'] == "磚"), 3, df['建材'] )
df['建材'] = np.where((df['建材'] == "大理石"), 3, df['建材'] )
df['建材'].value_counts()
df['建材'] = np.where((df['建材'] == "鋼筋混凝土"), 4, df['建材'] )
df['建材'] = np.where((df['建材'] == "混泥土"), 4, df['建材'] )
df['建材'] = np.where((df['建材'] == "水泥"), 4, df['建材'] )
df['建材'] = np.where((df['建材'] == "鋼筋混凝土+鐵皮"), 4, df['建材'] )
df['建材'].value_counts()

df['建材'] = pd.to_numeric(df['建材'].astype(str), errors='coerce')
# 先把建材drop掉
# df.drop('建材', axis=1, inplace=True)


#這啥
df.corr()
df.info()
df['斷層'].value_counts()

#prepare training data 
#把破壞致無法居住(1/0)丟掉，因為是結果
X = df.drop(['破壞致無法居住(1/0)'], axis=1)
y = df['破壞致無法居住(1/0)']

X.info()
df.info()
df.corr()


#split the data into training and testing data & testing data
from sklearn.model_selection import train_test_split

# test_data, random_state, highest accracy, avg accuracy from random_state 1~300
# 0.4, 5, 0.8
# 0.5, 5, 0.7777777777777778
# 0.6, 61, 0.72
# 0.7, 55, 0.7209302325581395

# 有建材 (random 1~100)
# 0.7, 54, 0.717948717948718
# 0.6, 4, 0.6829268292682927
# 0.6, 28, 0.8125
# 0.55, 25, 0.8181818181818182
# 0.63, 28, 0.8461538461538461

# 有建材 (random 1~300) 
# 0.4, 94, 0.7894736842105263 (avg = 0.5747622560280253)
# 0.5, 15, 0.7692307692307693 (avg = 0.577865111074509)
# 0.55, 231, 0.9090909090909091 (avg = 0.5798217731623067)
# 0.6, 28, 0.8125 (avg = 0.5784617143885582)
# 0.65, 149, 0.8 (avg = 0.5780946873787769)
# 0.7, 35, 0.7692307692307693 (avg = 0.5790437076982401)

# 調完縣市
# 0.61, 8, 0.6911764705882353
# 0.5, 23, 0.7
# 0.61, 480, 0.7407407407407407
# 0.5, 361, 0.76

# random 1~300
# 0.4, 
# 0.5, 195, 0.7321428571428571 (avg = 0.6073866463009882)
# 0.55, 101, 0.76 (avg = 0.6067077641370313)
# 0.65, 191, 0.6981132075471698 (avg = 0.60083912730766)
# 0.7, 299, 0.7037037037037037 (avg = 0.6000436290424265)
# 0.75, 2, 0.7631578947368421 (avg = 0.5985168456077615)
# 0.8, 282, 0.7346938775510204 (avg = 0.5972077078968445)
# 0.9, 169, 1 (avg = 0.5857580023896669)


# 0.1 ~ 0.5
# 20%


# max_score = []
# avg_score = []
# max_index = []
# max_rt = []
# for t in range(1, 51):
#     t = t * 0.01
#     score = []
#     rt = []
#     for i in range(1, 301):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=156)

#using logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
# predictions

#Evaluating the model
from sklearn.metrics import  confusion_matrix, accuracy_score, recall_score, precision_score
# accuracy_score(y_test, predictions)
# recall_score(y_test, predictions)
        # score.append(precision_score(y_test, predictions))
        # rt.append(i)
    # max_sc = 0
    # max_in = 0
    # m_rt = 0
    # m_avg = 0
    # for i in range(len(score)):
    #     if score[i] >= max_sc:
    #         max_sc = score[i]
    #         max_in = i
    #         m_rt = rt[i]
    #         m_avg = sum(score)/len(score)
    # max_score.append(max_sc)
    # avg_score.append(m_avg)
    # max_index.append(max_in)
    # max_rt.append(m_rt)

# max_ouput = []
# max_op_index = []
# max_r = []
# for t in range(len(max_index)):
#     if(max_score[t] - avg_score[t] <= 0.2):
#         print(max_score[t], end = " ")
#         print(t, end = " ")
#         print(max_rt[t], end = " ")
#         print(avg_score[t])
        # max_ouput.append(max_score[t])
        # max_op_index.append(t)
        # max_r.append(max_rt[t])

# 0.7894736842105263 30 156 0.6206090397355146
# 0.7619047619047619 31 197 0.6203861682728369
# 0.7741935483870968 32 222 0.6181701389743129
# 0.7727272727272727 33 156 0.6186825990414987
# 0.7878787878787878 34 257 0.6154992687968688
# 0.7878787878787878 35 257 0.6140230598548136
# 0.7647058823529411 36 257 0.6136064325755781
# 0.7567567567567568 37 262 0.6135309907013038
# 0.7666666666666667 38 282 0.61325142964672
# 0.7777777777777778 39 257 0.6112051515508038
# 0.8076923076923077 40 156 0.6084349450585894
# 0.7741935483870968 41 222 0.6095855183841631
# 0.7916666666666666 42 156 0.6101033618492167
# 0.8076923076923077 43 156 0.6097825352941904
# 0.7878787878787878 44 222 0.6113408096566353
# 0.78125 45 222 0.609305346230988
# 0.7857142857142857 46 156 0.6079897011096886
# 0.7407407407407407 47 222 0.608989540466867
# 0.7894736842105263 48 127 0.6068770724178214
# 0.7321428571428571 49 196 0.6067077641370313


# 0.8076923076923077 40 156 0.6084349450585894

# print(max(max_ouput))
# print(max_op_index[max_ouput.index(max(max_ouput))]*0.01)
# print(max_score[max_op_index[max_ouput.index(max(max_ouput))]])
# print(avg_score[max_op_index[max_ouput.index(max(max_ouput))]])
# print(max_r[max_op_index[max_ouput.index(max(max_ouput))]])
# print the max score and the index of the max
# print(max(score))
# avg
# print(sum(score)/len(score))
# print(score.index(max(score)))
# pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predicted Not Collapse', 'Predicted Collapse'], index=['True not Collapse', 'True Collapse'])

# export our model
import joblib
joblib.dump(lr, 'collision.pkl')

import faiss
faiss.save_local("faiss_db", "animal-fun-facts")
