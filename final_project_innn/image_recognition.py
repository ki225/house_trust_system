import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

# 載入已訓練的模型
model_inceptionV3 = tf.keras.models.load_model('model/Crack_Detection_InceptionV3_model.h5')
model_X_inceptionV3 = tf.keras.models.load_model('model/X-shaped_Crack_Detection_InceptionV3_model.h5')
model_Y_inceptionV3 = tf.keras.models.load_model('model/Y-shaped_Crack_Detection_InceptionV3_model.h5')

# 圖片辨識的函數
def predict_image(img_path):

    # 針對model_inceptionV3、model_X_inceptionV3調整圖片大小為150x150，並讀取處理圖片
    img = image.load_img(img_path, target_size=(150, 150)) # 調整目標大小為 150x150
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 正規化

    # 預測
    prediction = model_inceptionV3.predict(img_array)
    predictionX = model_X_inceptionV3.predict(img_array)
    
    # 針對model_Y_inceptionV3調整圖片大小為224x224，並讀取處理圖片
    img = image.load_img(img_path, target_size=(224, 224))  # 調整目標大小為 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 正規化
    
    # 預測
    predictionY = model_Y_inceptionV3.predict(img_array)
    
    if prediction[0] > 0.5:
        print(prediction[0])
        print("這張圖片被判定為有裂縫")       
        if predictionX[0] < 0.5:
            print(predictionX[0])
            print("這個裂縫被判定為X型裂縫")
        elif predictionY[0][0] < 0.5:
            print(predictionY[0][0])
            print("這個裂縫被判定為Y型裂縫")        
        else:
            print("這個裂縫是一般的裂縫")
    else:
        print(prediction[0])
        print("這張圖片被判定為沒有裂縫")

    # 顯示圖片
    plt.imshow(img)
    #plt.title('X型裂縫' if prediction[0] < 0.5 else '非X型裂縫')
    plt.show()

# 判斷是否有裂縫使用範例
img_path = 'IsCrackOrNot/crack/1.jpg'  
img_path = 'IsCrackOrNot/crack/2.jpg'  
img_path = 'IsCrackOrNot/crack/3.jpg'  
img_path = 'IsCrackOrNot/crack/4.jpg'  
img_path = 'IsCrackOrNot/not_crack/1.jpg'  
img_path = 'IsCrackOrNot/not_crack/2.jpg'  

# X型裂縫使用範例
img_path = 'IsXOrNot/valid/X/X1.jpg'  
img_path = 'IsXOrNot/valid/X/X2.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X1.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X2.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X3.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X4.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X5.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X6.jpg'  

# Y型裂縫使用範例
img_path = 'IsXOrNot/valid/X/X1.jpg'  
img_path = 'IsXOrNot/valid/X/X2.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X1.jpg'  
img_path = 'IsXOrNot/valid/not_X/not_X2.jpg'   
img_path = 'IsXOrNot/train/not_X/1.jpg'    

predict_image(img_path)