#!/usr/bin/env python

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu')) #(power of 2, common practice)
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/content/gdrive/My Drive/Experimental_Vocal_Images/Training_Set',
        target_size=(64, 64),
        # This should be same as input_shape.`
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/content/gdrive/My Drive/Experimental_Vocal_Images/Test_Set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


model.fit_generator(
        training_set,
        steps_per_epoch=160,
        epochs=25,
        validation_data=test_set,
        validation_steps=40)