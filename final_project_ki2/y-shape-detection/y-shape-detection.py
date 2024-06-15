import random
import numpy as np
import splitfolders
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
warnings.filterwarnings('ignore')

# 把 data/y-shape 資料夾裡的圖片分成訓練集、驗證集和測試集，並保留 neg/pos 分類
splitfolders.ratio('data/y-shape', output="data", seed=1337, ratio=(.7, 0.2,0.1)) 


# =============================== VGG16 ==============================================

IMAGE_SIZE = (224, 224)


train_path = './data/train'
val_path = './data/val'
test_path = './data/test'


train_datagen = ImageDataGenerator(rescale = 1./255)
eval_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('./data/train',
                                                 target_size = (224, 224), # should be same as the initialized one
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

val_generator = eval_datagen.flow_from_directory('./data/val',
                                                 target_size = (224, 224),
                                                 batch_size = 20,
                                                 class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory('./data/test',
                                                 target_size = (224, 224),
                                                 batch_size = 30,
                                                 class_mode = 'categorical')

# ================== VGG16 ==================
vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False 

last_output = vgg.output
norm = BatchNormalization()(last_output)
x = Dropout(0.5)(norm)
flat = Flatten()(x) 
pred = Dense(2, activation='softmax', name='softmax')(flat)
new_model = Model(inputs=vgg.input, outputs=pred)
new_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = new_model.fit(train_generator, epochs=10, validation_data=val_generator)

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('VGG16')
plt.show()






# test_image_files = test_generator.filepaths
# selected_images = random.sample(test_image_files, 5)

# for img_path in selected_images:
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     prediction = new_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
    
#     plt.imshow(img)
#     plt.title(f"Predicted class: {predicted_class[0]}")
#     plt.show()
#     print(f"Image: {img_path}")
#     print(f"Predicted class: {predicted_class[0]}")
#     print(f"Actual class: {img_path.split('/')[-2]}")

# ============= MobileNetV2 =============


mnv = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in mnv.layers:
    layer.trainable = False

mnv.summary() 
last_output = mnv.output
x = Flatten()(last_output)
pred = Dense(2, activation='softmax', name='softmax')(x)
MobileNetV2_model = Model(inputs=mnv.input, outputs=pred)
MobileNetV2_model.layers[-5:-1]
MobileNetV2_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
hist2 = MobileNetV2_model.fit(train_generator, epochs=10, validation_data=val_generator)


plt.figure(figsize=(12, 6))
plt.plot(hist2.history['accuracy'], label='Train Accuracy')
plt.plot(hist2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('MobileNetV2')
plt.show()


# test_image_files = test_generator.filepaths
# selected_images = random.sample(test_image_files, 5)

# for img_path in selected_images:
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     prediction = MobileNetV2_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
    
#     plt.imshow(img)
#     plt.title(f"Predicted class: {predicted_class[0]}")
#     plt.show()
#     print(f"Image: {img_path}")
#     print(f"Predicted class: {predicted_class[0]}")
#     print(f"Actual class: {img_path.split('/')[-2]}")

# =============================== ResNet50 ==============================================

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = False

resnet.summary()
last_output = resnet.output
x = Flatten()(last_output)
pred = Dense(2, activation='softmax', name='softmax')(x)
ResNet50_model = Model(inputs=resnet.input, outputs=pred)
ResNet50_model.summary()
ResNet50_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
hist3 = ResNet50_model.fit(train_generator, epochs=10, validation_data=val_generator)

plt.figure(figsize=(12, 6))
plt.plot(hist3.history['accuracy'], label='Train Accuracy')
plt.plot(hist3.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('ResNet50')
plt.show()


# test_image_files = test_generator.filepaths
# selected_images = random.sample(test_image_files, 5)

# for img_path in selected_images:
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     prediction = MobileNetV2_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
    
#     plt.imshow(img)
#     plt.title(f"Predicted class: {predicted_class[0]}")
#     plt.show()
#     print(f"Image: {img_path}")
#     print(f"Predicted class: {predicted_class[0]}")
#     print(f"Actual class: {img_path.split('/')[-2]}")


# =============================== InceptionV3 ==============================================
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inception = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

inception.summary()

last_output = inception.output
x = Flatten()(last_output)
pred = Dense(2, activation='softmax', name='softmax')(x)

InceptionV3_model = Model(inputs=inception.input, outputs=pred)
InceptionV3_model.summary()
InceptionV3_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
hist4 = InceptionV3_model.fit(train_generator, epochs=10, validation_data=val_generator)

plt.figure(figsize=(12, 6))
plt.plot(hist4.history['accuracy'], label='Train Accuracy')
plt.plot(hist4.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('InceptionV3')
plt.show()

# export model
import joblib
joblib.dump(hist4, 'InceptionV3_model.pkl')

# test_image_files = test_generator.filepaths
# selected_images = random.sample(test_image_files, 5)

# for img_path in selected_images:
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     prediction = MobileNetV2_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
    
#     plt.imshow(img)
#     plt.title(f"Predicted class: {predicted_class[0]}")
#     plt.show()
#     print(f"Image: {img_path}")
#     print(f"Predicted class: {predicted_class[0]}")
#     print(f"Actual class: {img_path.split('/')[-2]}")


# =============================== EfficientNetB0 ==============================================
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

efficientnet = EfficientNetB0(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
for layer in efficientnet.layers:
    layer.trainable = False
efficientnet.summary()
last_output = efficientnet.output
x = Flatten()(last_output)
pred = Dense(2, activation='softmax', name='softmax')(x)
EfficientNetB0_model = Model(inputs=efficientnet.input, outputs=pred)
EfficientNetB0_model.summary()
EfficientNetB0_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

hist5 = EfficientNetB0_model.fit(train_generator, epochs=10, validation_data=val_generator)

plt.figure(figsize=(12, 6))
plt.plot(hist5.history['accuracy'], label='Train Accuracy')
plt.plot(hist5.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('EfficientNetB0')
plt.show()


# test_image_files = test_generator.filepaths
# selected_images = random.sample(test_image_files, 5)

# for img_path in selected_images:
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     prediction = MobileNetV2_model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
    
#     plt.imshow(img)
#     plt.title(f"Predicted class: {predicted_class[0]}")
#     plt.show()
#     print(f"Image: {img_path}")
#     print(f"Predicted class: {predicted_class[0]}")
#     print(f"Actual class: {img_path.split('/')[-2]}")