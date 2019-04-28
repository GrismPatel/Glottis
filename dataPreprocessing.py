# First convert the images into tf.Records.
# Them use tf.Records to write tf.data
# !pip install tensorflow-gpu (uncomment this line on drive to install tf-gpu)

import tensorflow as tf
print (tf.__version__)
import glob as glob
import numpy as np
from PIL import Image
from google.colab import drive
import shutil
import traceback
import matplotlib.image as mpimg

drive.mount('/content/gdrive')

abnormals = '/content/gdrive/My Drive/Experimental_Vocal_Images/abnormal/'
tfrecord_filename = '/content/gdrive/My Drive/Experimental_Vocal_Images/abnormal/Experimental_Vocal_Images.tfrecords'

# create tf records
def _convert_image(img_path):
  label=0
  img_shape = mpimg.imread(img_path).shape
  filename = os.path.basename(img_path)
  
  with tf.gfile.GFile(img_path,'rb') as fid:
    image_data=fid.read()
    
  feature = {'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [3])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
            }
  
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  
  return example

def convert_image_folder(img_folder,tfrecord_file_name):
  img_paths = os.listdir(img_folder)
  img_paths = [os.path.abspath(os.path.join(img_folder,i))for i in img_paths]
  
  with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
    for img_path in img_paths[:101]:
      example =_convert_image(img_path)
      writer.write(example.SerializeToString())
      
convert_image_folder(abnormals, tfrecord_filename) 

# read the tf.records and convert them to tf.data.
def _extract_fn(tf_record):
  features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
  
  sample = tf.parse_single_example(tf_record,features)
  
  image = tf.image.decode_image(sample['image']) 
  img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
  label = sample['label']
  filename = sample['filename']
  
  return [image, label, filename, img_shape]

def extract_image():
  folder_path = '/content/gdrive/My Drive/Experimental_Vocal_Images/abnormal/ExtractedImages'
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
