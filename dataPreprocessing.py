# First convert the images into tf.Records.
# Them use tf.Records to write tf.data
# !pip install tensorflow-gpu (uncomment this line on drive to install tf-gpu)

import tensorflow as tf
print (tf.__version__)
import glob as glob
import numpy as np
from PIL import Image
from google.colab import drive

drive.mount('/content/gdrive')
abnormals = glob.glob('/content/gdrive/My Drive/Experimental_Vocal_Images/abnormal/*.jpg')

print (type(abnormals), len(abnormals))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


tfrecord_filename = '/content/gdrive/My Drive/Experimental_Vocal_Images/abnormal/Experimental_Vocal_Images.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

for my_gf in abnormals:
  img = Image.open(my_gf)
  label = 0
  
  feature = {'label': _int64_feature(label), 'image': _bytes_feature(img.tobytes())}
  
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example.SerializeToString())

writer.close()

# read the tf.records and convert them to tf.data.
# Pooja if you want/can then you can continue using this.