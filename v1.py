# Importing Libraries
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

kernel_initializer = tf.initializers.variance_scaling(scale= 2.0)
kernel_regularizer = tf.contrib.layers.l2_regularizer(scale= 1e-4)

image = Input(shape = (299, 299, 3))

layer1 = Conv2D(64, (1, 1),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(image)
layer1 = BatchNormalization(axis = 3)(layer1)

layer2 = Conv2D(96, (1, 1),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(image)
layer2 = BatchNormalization(axis = 3)(layer2)

layer21 = Conv2D(128, (3, 3),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(layer2)
layer21 = BatchNormalization(axis = 3)(layer21)

layer3 = Conv2D(16, (1, 1),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(image)
layer3 = BatchNormalization(axis = 3)(layer3)

layer31 = Conv2D(32, (5, 5),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(layer3)
layer31 = BatchNormalization(axis = 3)(layer31)

layer4 = MaxPooling2D((3, 3), padding = 'same', strides = (1, 1))(image)

layer41 = Conv2D(32, (1, 1),
              activation='relu',
              padding = 'same',
              kernel_initializer = kernel_initializer,
              kernel_regularizer = kernel_regularizer)(layer4)
layer41 = BatchNormalization(axis = 3)(layer41)

final = Concatenate(axis = 3)([layer1, layer21, layer31, layer41])

final = Flatten()(final)

final = Dense(units = 1, activation = 'softmax')(final)

finals = Model(input = [image], output = [final])
finals.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

plot_model(finals, to_file='v1.png', show_shapes=True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(299,299),
    batch_size=32,
    class_mode='binary')

validation_set = validation_datagen.flow_from_directory(
    'validation',
    target_size=(299,299),
    batch_size=32,
    class_mode='binary')

finals.fit_generator(
    training_set,
    steps_per_epoch=160,
    epochs=10,
    validation_data=validation_set,
    validation_steps=40)