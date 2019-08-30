#coding=utf-8
from pyspark import SparkConf,SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, concat_ws
import pyspark.sql.functions as f
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

import argparse
from pyspark.sql import types
import os

spark = SparkSession.builder.appName('tmp').getOrCreate()



parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="HDFS path to libsvm in parallelized format")
parser.add_argument("--date", help="date equal to input")
parser.add_argument("--output", help="s3 path to write result")

_args = parser.parse_args()

f = spark.read.csv(os.path.join(_args.datapath, _args.date))
#第一次解析
def modifyInput(row):
    row = row[0][1:-1]
    row = row.split(' ')
    row[0] = row[0][1:-1]
    row[1] = row[1][1:-1]
    row[2] = float(row[2][1:-1])
    return row

def modifyInput_v2(row):
    row0 = row[0][1:]
    row3 = row[3][:-1]
    row1 = row[1]
    row2 = row[2]
    out = (int(row0), int(row1), float(row2)*float(row3))
    return out

f = f.rdd.map(lambda row: modifyInput_v2(row)).map(lambda row: ((row[0]),[(row[2], row[1])])).reduceByKey(lambda x,y:x+y).\
    map(lambda row: (row[0], sorted(row[1], key=lambda text: text[0], reverse=True)))

def genJson(row):
    pid, matchIds = row[0], row[1]
    matchIds = matchIds[:200]
    matchIds = [id[1] for id in matchIds]
    result = dict()
    result.update({"item_key": {"s": str(pid)}})
    l = [{"n":str(id)} for id in matchIds]
    result.update({"info": {"l": l}})
    return result

f = f.map(lambda row: genJson(row))
f.saveAsTextFile(_args.output)



