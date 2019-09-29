import pyspark
from pyspark import SparkConf
import random
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from pyspark.sql.functions import concat, concat_ws, udf, shuffle, hash, broadcast, rand
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import col, when
from pyspark.sql import Row
import argparse
import json
import math
from pyspark.sql.functions import explode
import pyspark.sql.functions as F
import operator
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, desc, collect_list, row_number
from datetime import datetime, timedelta

_args_date = str(datetime.today()+timedelta(-1)).split(' ')[0]
def convertDate(date, delta):
    import datetime as dt
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date

item_track_path = 's3://jiayun.spark.data/wangqi/track/item/{date}/*'.format(date=_args_date)
user_track_path = 's3://jiayun.spark.data/wangqi/track/user/{date}/*'.format(date=_args_date)
out_path = 's3://jiayun.spark.data/wangqi/track/recommend_track/{date}/'.format(date=_args_date)

scences = ['SIMILARSORT']
blackCols = ['pid', 'pvid']

spark = SparkSession.builder.appName('recommend_track').getOrCreate()
sc = spark.sparkContext
random.seed(777)

#------------------------------User Track Json Generate For each scenes
userdf = spark.read.json(user_track_path)#.drop(*blackCols)
for scen in scences:
    scenRDD = userdf.filter(col('scences') == scen).rdd.\
        map(lambda row: {k: v for k, v in row.asDict().items() if (v is not None) and (k != 'scences')})
    scenRDD.saveAsTextFile(out_path+'user/'+scen)

#------------------------------Item Track Json Generate For each scenes
itemdf = spark.read.json(item_track_path)#.drop(*blackCols)
for scen in scences:
    scenRDD = itemdf.filter(col(scen).isNotNull()).select('{}.*'.format(scen), 'pvid').rdd.\
        map(lambda row: {k: v for k, v in row.asDict().items() if (v is not None) and (k != 'scences')})
    scenRDD.saveAsTextFile(out_path+'item/'+scen)

