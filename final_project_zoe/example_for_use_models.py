from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# load model
model = load_model('h5 file name')

def preprocess_image(img_path):
    """
    preprocess image
    :param img_path: image path
    :return: preprocessed image
    """
    # load image and resize to 150x150
    img = image.load_img(img_path, target_size=(150, 150))
    # convert image to array
    img_array = image.img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img_array = np.expand_dims(img_array, axis=0)
    # scale the image pixels to [0,1]
    img_array /= 255.0
    return img_array

def show_image(img_path):
    """
    show image
    :param img_path: image path
    """
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def predict_crack(model, img_path):
    """
    predict whether the image contains a crack
    :param model: model
    :param img_path: image path
    """
    # preprocess image
    img_array = preprocess_image(img_path)
    # predict the image
    prediction = model.predict(img_array)
    # print prediction
    print(f'Prediction for {img_path}:', prediction)
    
    if prediction > 0.5:
        print("Prediction: Crack")
    else:
        print("Prediction: No Crack")
    
    # show image
    show_image(img_path)

# test the model
img_path = 'your picture file name'  # path to the image
predict_crack(model, img_path)