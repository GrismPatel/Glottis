#!/usr/bin/env python

import json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

with open('/Users/grism/Desktop/learning/Glottis/sequential_config.json') as json_file:
    data = json.load(json_file)

print (data)
model = Sequential()

first_layer = data['first_layer']
print ('First Layer: ', first_layer)
model.add(Conv2D(data[first_layer]['number_kernels'],
                 (data[first_layer]['kernel_width'], data[first_layer]['kernel_height']),
                 activation = data[first_layer]['activation'],
                 kernel_initializer = tf.initializers.variance_scaling(scale= data[first_layer]['kernel_initializer_scale']),
                 kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[first_layer]['kernel_regularizer_scale'])))

second_layer = data[first_layer]['next']
print ('Second Layer: ', second_layer)
model.add(MaxPool2D(pool_size = (data[second_layer]['pool_width'], data[second_layer]['pool_height'])))

third_layer = data[second_layer]['next']
print ('Third Layer: ', third_layer)
model.add(Conv2D(data[third_layer]['number_kernels'],
                 (data[third_layer]['kernel_width'], data[third_layer]['kernel_height']),
                 activation = data[third_layer]['activation'],
                 kernel_initializer = tf.initializers.variance_scaling(scale= data[third_layer]['kernel_initializer_scale']),
                 kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[third_layer]['kernel_regularizer_scale'])))

fourth_layer = data[third_layer]['next']
print ('Fourth Layer: ', fourth_layer)
model.add(MaxPool2D(pool_size = (data[fourth_layer]['pool_width'], data[fourth_layer]['pool_height'])))

fifth_layer = data[fourth_layer]['next']
print ('Fifth Layer: ', fifth_layer)
model.add(Flatten())

sixth_layer = data[fifth_layer]['next']
print ('Sixth Layer: ', sixth_layer)
model.add(Dense(units = data[sixth_layer]['units'],
                activation = data[sixth_layer]['activation']))

seventh_layer = data[sixth_layer]['next']
print ('Seventh Layer: ', seventh_layer)
model.add(Dense(units = data[seventh_layer]['units'],
                activation = data[seventh_layer]['activation']))

model.compile(optimizer = data['compile']['optimizer'], loss = data['compile']['loss'], metrics = ['accuracy'])

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