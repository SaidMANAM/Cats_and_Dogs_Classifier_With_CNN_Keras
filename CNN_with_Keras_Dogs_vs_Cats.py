from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import random
import pandas as pd
import numpy as np
import sys
from PIL import Image
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame, Series
from pandas.core.arrays import ExtensionArray
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau


################################################

TRAINING_DIR = "/home/said/Téléchargements/22535_28903_bundle_archive/dataset/training_set"
train_datagen=ImageDataGenerator(rescale=1.0/255.0,
rotation_range=40, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.2,zoom_range=0.2,horizontal_flip=True,
                                 fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=250,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/home/said/Téléchargements/22535_28903_bundle_archive/dataset/test_set"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=250,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

tr=len(os.listdir('/home/said/Téléchargements/22535_28903_bundle_archive/dataset/training_set/cats'))+len(os.listdir('/home/said/Téléchargements/22535_28903_bundle_archive/dataset/training_set/dogs'))
te=len(os.listdir('/home/said/Téléchargements/22535_28903_bundle_archive/dataset/test_set/cats'))+len(os.listdir('/home/said/Téléchargements/22535_28903_bundle_archive/dataset/test_set/dogs'))
print("te=",te)
print("tr=",tr)

##################################### Callback #############################
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            self.model.stop_training=True
##################################################  MODEL  ###################################
call=mycallback( )

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', ),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=40, steps_per_epoch=32,
                    validation_data=validation_generator, validation_steps=8,callbacks=[call])

############################### plotting accuracy and loss #####################################
import matplotlib
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


