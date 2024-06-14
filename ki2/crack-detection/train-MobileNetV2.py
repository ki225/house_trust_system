import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import splitfolders
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
splitfolders.ratio('data/cracks', output="data", seed=1337, ratio=(.7, 0.2,0.1)) 

# ==============================

IMAGE_SIZE = (224, 224)

train_path = './data/train'
val_path = './data/val'
test_path = './data/test'

train_datagen = ImageDataGenerator(rescale = 1./255)
eval_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_batch = train_datagen.flow_from_directory('./data/train',
                                                 target_size = (224, 224), # should be same as the initialized one
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

val_batch = eval_datagen.flow_from_directory('./data/val',
                                                 target_size = (224, 224),
                                                 batch_size = 20,
                                                 class_mode = 'categorical')

test_batch = test_datagen.flow_from_directory('./data/test',
                                                 target_size = (224, 224),
                                                 batch_size = 30,
                                                 class_mode = 'categorical')


# get the class names
class_names = train_batch.class_indices
print(class_names)


vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False 

vgg.summary() 


# we create our own last layers
last_output = vgg.output

norm = BatchNormalization()(last_output)
x = Dropout(0.5)(norm)

flat = Flatten()(x) 
pred = Dense(2, activation='softmax', name='softmax')(flat) # we have 2 possibilites: Positive or Negative

new_model = Model(inputs=vgg.input, outputs=pred) 
new_model.layers[-5:-1]


new_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', 
 metrics=['accuracy'])

start_time = time.perf_counter()

hist1 = new_model.fit(train_batch, steps_per_epoch=18,
             validation_data=val_batch, validation_steps=3, epochs=5,
             verbose=1)

time1 = round(time.perf_counter() - start_time, 2)
print (f'\n\nTime taken by VGG16: {time1} seconds')


# ===============================================

mnv = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in mnv.layers:
    layer.trainable = False

mnv.summary() 


last_output = mnv.output

x = Flatten()(last_output)
pred = Dense(2, activation='softmax', name='softmax')(x)

new_model = Model(inputs=mnv.input, outputs=pred)

new_model.layers[-5:-1]

new_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', 
 metrics=['accuracy'])


start_time = time.perf_counter()

hist2 = new_model.fit(train_batch, steps_per_epoch=18,
             validation_data=val_batch, validation_steps=3, epochs=5,
             verbose=1)

time2 = round(time.perf_counter() - start_time, 2)
print (f'\n\nTime taken by MobileNetV2: {time2} seconds')


plt.figure(figsize=(10, 10))
for images, labels in test_batch.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")



#model export
import joblib
joblib.dump(vgg, 'model/model.pkl')
