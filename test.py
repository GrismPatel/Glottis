#!/usr/bin/env python

import os
import json
import tensorflow as tf
import datetime

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from shutil import copyfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
path = os.getcwd()

with open(path + '/Glottis/sequential_config.json') as json_file:
    data = json.load(json_file)

model = Sequential()

try:
    first_layer = data['first_layer']
except KeyError:
    raise KeyError ('config does not contain first_layer. Please add it.')

try:
    data[first_layer]
except KeyError:
    raise KeyError ('first layer object is missing in the config. Please add it.')

try:
    layer_type = data[first_layer]['type']
except KeyError:
    raise KeyError ('first layer config type is missing. Please add it.')

print ('Current Layer is {layer} of type: {layer_type}'.format(layer_type = layer_type, layer = first_layer))
model.add(Conv2D(data[first_layer]['number_kernels'],
                 (data[first_layer]['kernel_width'], data[first_layer]['kernel_height']),
                 input_shape = (data['image_width'], data['image_height'], data['channels']),
                 activation = data[first_layer]['activation'],
                 kernel_initializer = tf.initializers.variance_scaling(scale= data[first_layer]['kernel_initializer_scale']),
                 kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[first_layer]['kernel_regularizer_scale'])))

current_layer = data[first_layer]['next']

while True:
    try:
        data[current_layer]
    except KeyError:
        raise KeyError ('{current_layer} object is not defined in the config. Please add it'.format(current_layer = current_layer))

    try:
        layer_type = data[current_layer]['type']
    except KeyError:
        raise KeyError ('{current_layer} object has no type in the config. Please add it.'.format(current_layer = current_layer))

    print ('Current Layer is {layer} of type: {layer_type}'.format(layer_type = layer_type, layer = current_layer))
    
    if layer_type == 'conv2d':
        model.add(Conv2D(data[current_layer]['number_kernels'],
                         (data[current_layer]['kernel_width'], data[current_layer]['kernel_height']),
                         activation = data[current_layer]['activation'],
                         kernel_initializer = tf.initializers.variance_scaling(scale= data[current_layer]['kernel_initializer_scale']),
                         kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[current_layer]['kernel_regularizer_scale'])))
    
    elif layer_type == 'maxpool2d':
        model.add(MaxPool2D(pool_size = (data[current_layer]['pool_width'], data[current_layer]['pool_height'])))

    elif layer_type == 'dense':
        model.add(Dense(units = data[current_layer]['units'],
                        activation = data[current_layer]['activation']))

    elif layer_type == 'flatten':
        model.add(Flatten())

    else:
        raise ValueError ('Invalid layer type in the config. layer_type {layer_type} is not defined in the code.'.format(layer_type = layer_type))
    
    current_layer = data[current_layer]['next']
    
    if current_layer == None:
        break
    
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

result_path = path + '/Glottis/Results/{time}/'.format(time = datetime.datetime.now())

try:
    os.makedirs(result_path)
except OSError:
    raise OSError ('Failed to create path')

plot_model(model, to_file = result_path + 'GlottisCNN.png', show_shapes = True)
print ('Model saved successfully to path: {result_path}GlottisCNN.png'.format(result_path = result_path))

csv_logger = CSVLogger(result_path + 'GlottisCNN.csv', append=True, separator='|')
print ('Logs saved successfully to path: {result_path}GlottisCNN.csv'.format(result_path = result_path))

copyfile(path + '/Glottis/sequential_config.json', result_path + 'config.json')
print ('Copied the config sucessfully to path: {result_path}config.json').format(result_path = result_path)

model.fit_generator(
        training_set,
        steps_per_epoch=160,
        epochs=2,
        validation_data=test_set,
        validation_steps=40)