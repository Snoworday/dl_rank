

name_list = ['itemid', 'cid', 'cross', 'gmv', 'ctr']
label_path = ['s3://jiayun.spark.data/wangqi/wide_deep/Feature/itemid/2019-09-03/part-00000-93ea99ba-05d6-4de9-a554-d2207e443431-c000.txt',
             's3://jiayun.spark.data/wangqi/wide_deep/Feature/cid/2019-09-03/part-00000-58e046e0-fe70-4673-899a-78fb71091d4c-c000.txt',
             's3://jiayun.spark.data/wangqi/wide_deep/Feature/idpair/2019-09-03/part-00000-9face73f-c5f1-4f6f-ae98-5370c41b2bac-c000.txt',
               3000, 3000]
value_path = ['s3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/deep_layer_variable/pid_embedding.txt',
              's3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/wide_layer_variable/cid_embedding.txt',
              's3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/wide_layer_variable/click_cross_embedding.txt',
              's3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/wide_layer_variable/gmv_embedding.txt',
              's3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/wide_layer_variable/ctr_embedding.txt']



# hadoop fs -cp s3://jiayun.spark.data/wangqi/wide_deep/v1/online_graph/txt/wide_layer_variable/ctr_embedding.txt /user/hadoop
# hadoop fs -copyToLocal /user/hadoop/ctr_embedding.txt

import pyspark
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from functools import reduce
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

spark = SparkSession.builder.appName('tmp2').getOrCreate()

out = []


def concat(key):
    @udf
    def wrapper(l):
        return key+'_'+l
    return wrapper

merge_list = []
for key, label_path_, value_path_ in zip(name_list[1:], label_path[1:], value_path[1:]):
    if isinstance(label_path_, str):
        label = spark.read.text(label_path_)
        new_row = spark.createDataFrame([['DEFAULT']])
        label = new_row.union(label).withColumn('label', concat(key)('_1')).drop('_1')
        label = label.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

        value = spark.read.text(value_path_).withColumnRenamed('_1', 'value')
        value = value.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

        merge = label.join(value, label.id==value.id, 'inner').drop('id')
        merge_list.append(merge)

out = reduce(lambda x, y: x.union(y), merge_list)
out.withColumn('merge', concat_ws(',', out['label'], out['value'])).select('merge')\
    .write.text('s3://jiayun.spark.data/wangqi/wide_deep/online/data/wide')


merge_list = []
for key, label_path_, value_path_ in zip(name_list[:1], label_path[:1], value_path[:1]):
    if isinstance(label_path_, str):
        label = spark.read.text(label_path_)
        new_row = spark.createDataFrame([['DEFAULT']])
        label = new_row.union(label).withColumn('label', concat(key)('_1')).drop('_1')
        label = label.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

        value = spark.read.text(value_path_).withColumnRenamed('_1', 'value')
        value = value.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

        merge = label.join(value, label.id==value.id, 'inner').drop('id')
        merge_list.append(merge)

out = reduce(lambda x, y: x.union(y), merge_list)
out.withColumn('merge', concat_ws(',', out['label'], out['value'])).select('merge')\
    .write.text('s3://jiayun.spark.data/wangqi/wide_deep/online/data/deep')
