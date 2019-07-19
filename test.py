#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

with open('/Users/grism/Desktop/learning/Glottis/sequential_config.json') as json_file:
    data = json.load(json_file)

print (data)
model = Sequential()

first_layer = data['first_layer']
layer_type = data[first_layer]['type']

print ('Current Layer is {layer} of type: {layer_type}'.format(layer_type = layer_type, layer = first_layer))
model.add(Conv2D(data[first_layer]['number_kernels'],
                 (data[first_layer]['kernel_width'], data[first_layer]['kernel_height']),
                 input_shape = (data['image_width'], data['image_height'], data['channels']),
                 activation = data[first_layer]['activation'],
                 kernel_initializer = tf.initializers.variance_scaling(scale= data[first_layer]['kernel_initializer_scale']),
                 kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[first_layer]['kernel_regularizer_scale'])))

current_layer = data[first_layer]['next']
while True:

    layer_type = data[current_layer]['type']
    print ('Current Layer is {layer} of type: {layer_type}'.format(layer_type = layer_type, layer = current_layer))
    if layer_type == 'conv2d':
        model.add(Conv2D(data[current_layer]['number_kernels'],
                         (data[current_layer]['kernel_width'], data[current_layer]['kernel_height']),
                         activation = data[current_layer]['activation'],
                         kernel_initializer = tf.initializers.variance_scaling(scale= data[current_layer]['kernel_initializer_scale']),
                         kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= data[current_layer]['kernel_regularizer_scale'])))
    
    if layer_type == 'maxpool2d':
        model.add(MaxPool2D(pool_size = (data[current_layer]['pool_width'], data[current_layer]['pool_height'])))

    if layer_type == 'dense':
        model.add(Dense(units = data[current_layer]['units'],
                        activation = data[current_layer]['activation']))


    if layer_type == 'flatten':
        model.add(Flatten())

    current_layer = data[current_layer]['next']
    
    if current_layer == None:
        break
    
model.compile(optimizer = data['compile']['optimizer'], loss = data['compile']['loss'], metrics = ['accuracy'])
plot_model(model, to_file="CNN_Grism.png", show_shapes=True)

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