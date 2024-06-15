from flask import Flask, request, jsonify
import pandas as pd 
import numpy as np
import joblib
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from configparser import ConfigParser

# Config Parser
config = ConfigParser()
config.read("config.ini")

#把密碼寫到環境變數
os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-pro")

house_model = joblib.load('collision.pkl')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FIXED_FILENAME = 'crack.png'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return app.send_static_file('index.html')

# 處理縣市輸入
def city_data(s):
    city5_list = ["花蓮縣"]
    city4_list = ["台東縣", "臺東縣"]
    city3_list = ["台中市", "台南市", "南投縣", "臺中市", "臺南市"]
    city2_list = ["台北市", "桃園市",  "新竹市", "新竹縣", "彰化縣", "臺北市"]
    city1_list = [ "新北市", "高雄市", "基隆市", "嘉義市", "苗栗縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣"] 
    for city in city5_list:
        if s in city:
            return 5
    for city in city4_list:
        if s in city:
            return 4
    for city in city3_list:
        if s in city:
            return 3
    for city in city2_list:
        if s in city:
            return 2
    for city in city1_list:
        if s in city:
            return 1
    return 2

# 處理建材輸入
def material_data(s):
    material0_list = ["沙拉油桶、報紙、紙袋混充", "豆腐渣"]
    material1_list = ["鐵皮", "無筋磚砌體（無地基）", "土"]
    material2_list = ["洗石", "檜木", "木材", "石塊", "磚木"]
    material3_list = ["紅磚","磚瓦", "磚", "大理石"]
    material4_list = ["鋼筋混凝土", "混泥土", "水泥", "鋼筋混凝土+鐵皮", "鋼筋混土"]
    for material in material0_list:
        if s in material:
            return 0
    for material in material1_list:
        if s in material:
            return 1
    for material in material2_list:
        if s in material:
            return 2
    for material in material3_list:
        if s in material:
            return 3
    for material in material4_list:
        if s in material:
            return 4
    return 3

# 右邊bot
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()

    # user_mess 的type 是str
    user_message = data['message']
    if "我要預測" in user_message:
        reply = f"請在左邊表單輸入要預測的建築物資訊啾咪"
        return jsonify({'reply': reply})
    
    user_message += " ，請使用繁體中文回答。"
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": user_message,
            }
        ]
    )
    result = llm.invoke([message])
    reply = result.content
    return jsonify({'reply': reply})

# 左邊表單
@app.route('/submit', methods=['POST'])
def send_info():
    if request.method == "POST":
        form_data = request.form
        # city
        City_ = 0
        City_ = city_data(form_data["City"])
        

        # 建材
        Material_ = 0
        Material_ = material_data(form_data["Material"])
        
        data = [
            [
                City_,
                int(form_data["Fault"]),
                int(form_data["Soil_Liquefaction"]),
                int(form_data["Land_Subside"]),
                Material_,
                float(form_data["Floor"]),
            ]
        ]
        
        data = np.array(data).reshape(1, -1)
        result = house_model.predict(data)

        # 預測結果
        ans = ""
        if result[0] == 0:
            ans = "大概不會倒"
        else:
            ans = "有可能會倒喔"

    # 圖片處理
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']

    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], FIXED_FILENAME)
        file.save(file_path)
        return app.send_static_file('index.html')
    return 'File type not allowed'
    
if __name__ == '__main__':
    app.run(debug=True)
