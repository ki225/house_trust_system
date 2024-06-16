from keras.applications.inception_v3 import InceptionV3
from keras import Model, layers
import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights='imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = train_datagen.flow_from_directory('IsXOrNot',
                                                    target_size=(150, 150),
                                                    batch_size=64,
                                                    shuffle=True,
                                                    class_mode='binary',
                                                    subset='training')

print(train_generator.class_indices)
print(train_generator.classes)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
validation_generator = validation_datagen.flow_from_directory('IsXOrNot',
                                                              target_size=(
                                                                  150, 150),
                                                              batch_size=64,
                                                              class_mode='binary',
                                                              subset='validation')

print(validation_generator.class_indices)
print(validation_generator.classes)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=7,
    verbose=1,
    callbacks=[callbacks],
)

# 訓練過程中準確度和損失的歷史記錄
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 設置 X 軸的範圍，從 1 開始
epochs = range(1, len(acc) + 1)

# 繪製訓練和驗證準確度
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)  # 設置 X 軸刻度
plt.legend()  # 顯示圖例

plt.figure()

# 繪製訓練和驗證損失
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)  # 設置 X 軸刻度
plt.legend()  # 顯示圖例

plt.show()

model.save('Crack_Detection_InceptionV3_model.h5')