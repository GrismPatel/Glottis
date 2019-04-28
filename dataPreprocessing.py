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
import shutil

def _extract_fn(tf_record):
  features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
  
  sample=tf.parse_single_example(tf_record,features)
  
  image = tf.image.decode_image(sample['image']) 
  img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
  label = sample['label']
  filename = sample['filename']
  
  return [image, label, filename, img_shape]

import traceback

def extract_image():
  folder_path = '/content/gdrive/My Drive/ExtractedImages'
  shutil.rmtree(folder_path, ignore_errors = True)
  os.mkdir(folder_path)
  
  dataset = tf.data.TFRecordDataset([tfrecord_filename])
  dataset = dataset.map(_extract_fn)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  
  with tf.Session()  as sess:
    sess.run(tf.global_variables_initializer())
    
    try:
      for i in range(101):
        image_data = sess.run(next_element)
                
        if not np.array_equal(image_data[0].shape, image_data[3]):
          print('Image {} not decoded properly'.format(image_data[2]))
          continue
              
        save_path = os.path.abspath(
            os.path.join(folder_path, image_data[2].decode('utf-8')))
        
        mpimg.imsave(save_path, image_data[0])
        print('Save path= ', save_path, 'Label= ', image_data[1])
      
    except Exception as e: 
      traceback.print_exc()
      print('Except: ',e)