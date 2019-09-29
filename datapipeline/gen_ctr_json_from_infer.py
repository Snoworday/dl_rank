#coding=utf-8
from pyspark import SparkConf,SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, concat_ws
import pyspark.sql.functions as f
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import datetime
import argparse
import sys
from pyspark.sql import types
import os
from tensorflow import gfile



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

def modifyInput_v3(row):
    row0, row1, row2, row3 = row.split(',')
    out = (int(row0), int(row1), float(row2)*float(row3))
    return out



def genJson(row):
    pid, matchIds = row[0], row[1]
    matchIds = matchIds[:200]
    matchIds = [id[1] for id in matchIds]
    result = dict()
    result.update({"trigger_key": {"s": str(pid)}})
    l = [{"n":str(id)} for id in matchIds]
    result.update({"pairs": {"l": l}})
    return result

def genJson_v2(row):
    pid, matchs = row[0], row[1]
    matchs = matchs[:200]
    Scores = [item[0] for item in matchs]
    Ids = [item[1] for item in matchs]
    pairs = [{"m":{"key":{"s":str(key)}, 'score':{"n":str(score)}}} for key,score in zip(Ids, Scores)]
    result = dict()
    result.update({"pairs": {"l":pairs}})
    result.update({'trigger_key': {"s":str(pid)}})
    return result

spark = SparkSession.builder.appName('tmp').getOrCreate()
sc = spark.sparkContext

parser = argparse.ArgumentParser()
_args = parser.parse_args()
_args.date = str(datetime.datetime.today()).split(' ')[0]
_args.datapath = 's3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/result/{}'.format(_args.date)
_args.output = 's3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/resultJson/{}'.format(_args.date)

if gfile.Exists(_args.output):
    gfile.DeleteRecursively(_args.output)

f = sc.textFile(_args.datapath)
f = f.map(lambda row: modifyInput_v3(row)).map(lambda row: ((row[0]), [(row[2], row[1])])).reduceByKey(
    lambda x, y: x + y). \
    map(lambda row: (row[0], sorted(row[1], key=lambda text: text[0], reverse=True)))
f = f.map(lambda row: genJson_v2(row))
f.saveAsTextFile(_args.output)



