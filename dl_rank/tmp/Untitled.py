#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tmp.tmp2


import numpy as np
import os
import pandas as pd
import shutil
import tarfile
import time
import zipfile

import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import ArrayType, FloatType
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from pyspark.sql.types import *

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

import tensorflow as tf
tf.__version__


# In[2]:

conf = SparkConf().set(key="spark.jars", value="/Users/snoworday/git/ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.13.1.jar")
data_url = "http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz"
# Directory to store the downloaded data.
data_dir = "/Users/snoworday/git/ecosystem/spark/spark-tensorflow-connector/resnet/"
tensorflow_graph_dir = data_dir + 'resnet_v2_fp32_savedmodel_NCHW/1538687196/'

# spark = SparkSession \
#     .builder \
#     .appName('DeepFM') \
#     .getOrCreate()

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# In[3]:


def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urlretrieve(url=url, filename=file_path)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


# In[5]:


maybe_download_and_extract(url=data_url, download_dir=data_dir)


# In[6]:


sess = tf.compat.v1.Session()
model = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], tensorflow_graph_dir)
bc_model = sc.broadcast(model)
sess.close()


# In[7]:


input_local_dir = "/tmp/data/flowers"


# In[8]:


from pyspark.sql.types import *

schema = StructType([StructField('image/class/label', IntegerType(), True),
                     StructField('image/width', IntegerType(), True),
                     StructField('image/height', IntegerType(), True),
                     StructField('image/format', StringType(), True),
                     StructField('image/encoded', BinaryType(), True)])




def parse_example(image_data):

  image = tf.image.decode_jpeg(image_data, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.central_crop(image, central_fraction=0.875)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [224, 224],
                                     align_corners=False)
  image = tf.squeeze(image, [0])
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def predict_batch(image_batch):
  batch_size = len(image_batch)
  sess = tf.compat.v1.Session()

  batch_size = 64
  image_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
  dataset = tf.data.Dataset.from_tensor_slices(image_input)
  dataset = dataset.map(parse_example, num_parallel_calls=16).prefetch(512).batch(batch_size)
  iterator = dataset.make_initializable_iterator()
  image = iterator.get_next()

  tf.compat.v1.train.import_meta_graph(bc_model.value)
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(iterator.initializer, feed_dict={image_input: image_batch})
  softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')
  result = []
  try:
    while True:
      batch = sess.run(image)
      preds = sess.run(softmax_tensor, {'input_tensor:0': batch})
      result = result + list(preds)
  except tf.errors.OutOfRangeError:
    pass

  return pd.Series(result)


df = spark.read.format("tfrecords").schema(schema).load(input_local_dir+'/flowers_train*.tfrecord')
df = df.limit(3200)


# In[ ]:



# image_batch = df.limit(128).toPandas().loc[: , "image/encoded"].apply(lambda x: bytes(x))
# images = predict_batch(image_batch)
# print(images.shape)
predict_batch_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)
predictions = df.select(predict_batch_udf(col("image/encoded")).alias("prediction"))
predictions.write.mode("overwrite").save("/tmp/predictions")
result_df = spark.read.load("/tmp/predictions")
display(result_df)