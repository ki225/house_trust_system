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

import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        if len(images) > 3:
            break
    fig = plt.figure(figsize=(10, 12))
    xrange = range(1, 5)

    for img, x in zip(images, xrange):
        ax = fig.add_subplot(2, 2, x)
        ax.imshow(img)
        ax.set_title(img.shape)

# data is on the hackmd's link
load_images_from_folder("../archive/Positive")
load_images_from_folder("../archive/Negative")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' #Please Google it and download it(It is too big to upload)
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False

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

train_generator = train_datagen.flow_from_directory('../archive',
                                                    target_size=(150, 150),
                                                    batch_size=64,
                                                    shuffle=True,
                                                    class_mode='binary',
                                                    subset='training')

print(train_generator.class_indices)
print(train_generator.classes)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

validation_generator = validation_datagen.flow_from_directory('../archive',
                                                              target_size=(
                                                                  150, 150),
                                                              batch_size=64,
                                                              class_mode='binary',
                                                              subset='validation')

print(validation_generator.class_indices)
print(validation_generator.classes)

callbacks = myCallback()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=7,
    verbose=1,
    callbacks=[callbacks],
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')

plt.legend()

plt.show()

model.save('Crack_Detection_InceptionV3_model.h5')
