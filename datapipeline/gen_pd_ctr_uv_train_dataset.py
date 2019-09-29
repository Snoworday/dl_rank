from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, concat_ws, rand
import pyspark.sql.functions as f
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import logging
import argparse
import yaml
from datetime import datetime, timedelta

# from util.data_info import item_profile_head_4_1 as head
DEBUG = False
if DEBUG:
    with open('separator.yaml', 'r') as f:
        sparator = yaml.load(f)
        primary_delim = sparator['primary_delim']
        secondary_delim = sparator['secondary_delim']
        teriary_delim = sparator['teriary_delim']
else:
    primary_delim = '@'
    secondary_delim = '|'
    teriary_delim = '&'



_args_date = str(datetime.today()+timedelta(-1)).split(' ')[0]

spark = SparkSession.builder.appName('merge_label_itemProfile').getOrCreate()

train = True
ratioSample = 0.1#_
# cate_voc = ['catid1','catid2','catid3']
# ignore=['log_date','pid','pno','is_sale','sku_num','gmv_1d','gmv_7d','gmv_15d','gmv_30d','gmv_60d','gmv_90d','gmv']




def convertDate(date, delta):
    import datetime as dt
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date

if DEBUG:
    labelPath = "s3://jiayun.spark.data/wangqi/I2IRankData/TextFile/LabelData"
    itemProfilePath = "s3://jiayun.spark.data/songfeng/product/swing_time1/headHold_5-rankHold_400-dayHold1_72-dayHold2_360-tailHold_700/cate_sort/csv/2019-07-01"
    predDFOutPath = 's3://jiayun.spark.data/wangqi/I2IRankData/TextFile/PredDemoData'
else:
    if train:
        labelPath = 's3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/label_ctr_uv/{}'.format(_args_date)
        trainDFOutPath = "s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/traindata/{}".format(_args_date)
        evalDFOutPath = "s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/evaldata/{}".format(_args_date)
    else:
        labelPath = 's3://jiayun.spark.data/songfeng/test/recall/i2i/sort/swing_catehotAll_India/result/csv/'
        # labelPath = 's3://jiayun.spark.data/wangqi//I2IRankData/TextFile/LabelData/'
        predDFOutPath = 's3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/preddata/{}'.format(_args_date)
    itemProfilePath = "s3://jiayun.spark.data/product_algorithm/product_statistics_features/item_profile_union/{}/".format(convertDate(_args_date, -2))

colFactory = udf(lambda x: x.split('|')[1])
def checklabel(X):
    item1, item2, ctrlabel, uvlabel = X
    item1 = item1.replace(" ",'')
    item2 = item2.replace(" ",'')
    ctrlabel = ctrlabel.replace(" ",'')
    uvlabel = uvlabel.replace(" ", '')
    if item1 == "" or item1.replace(" ", '') == "":
        return False
    if item2 == "" or item2.replace(" ", '') == "":
        return False
    if ctrlabel == "" or ctrlabel.replace(" ", '') == "":
        return False
    if uvlabel == "" or uvlabel.replace(" ", '') == "":
        return False
    return True

def parseItemProfile(X):
    rawdata = X[0].split('|')
    return (rawdata[1], str(X))


def mergeTwoItemProfile(x, y):
    xs = x.split('|')
    ys = y.split('|')
    merge = list(zip(xs, ys))
    merge_t = [teriary_delim.join(attr) for attr in merge]
    merge_s = secondary_delim.join(merge_t)
    return merge_s
u_mergeTwoItemProfile = udf(mergeTwoItemProfile)

#read profile
itemProfileDF = spark.read.text(itemProfilePath).rdd.repartition(1000).filter(lambda x:x[0].split('|')[1] != 'pid').toDF()
itemProfileDF = itemProfileDF.withColumn('pid', colFactory(itemProfileDF.value))
#read label
if train:
    rawLabelDF = spark.read.text(labelPath).rdd.repartition(5000).map(lambda input_: input_.value.split('|')).filter(
        lambda input_: len(input_) == 4).filter(checklabel).toDF().selectExpr("_1 as item1", "_2 as item2",
                                                                              "_3 as ctr_label", "_4 as uv_label")
    recordCount = rawLabelDF.cache().count()
    testCount = recordCount//10
    trainCount = recordCount - testCount
    # read label

    rawLabelPositiveDF = rawLabelDF.filter(rawLabelDF[2] == 1)
    rawLabelNegativeDF = rawLabelDF.filter(rawLabelDF[2] == 0)
    PCount = rawLabelPositiveDF.count()
    NCount = rawLabelNegativeDF.count()
    ratio = PCount/NCount
    assert ratio<1 and ratio>0, 'ratio should be in (0, 1),but now {}'.format(ratio)
    rawLabelNegativeDF_sample = rawLabelNegativeDF.sample(withReplacement=False, fraction=PCount/NCount)
    LabelDF_sample = rawLabelPositiveDF.union(rawLabelNegativeDF_sample)
    LabelDF = rawLabelPositiveDF.union(rawLabelNegativeDF)
else:
    rawLabelDF = spark.read.text(labelPath).rdd.repartition(1000).map(
        lambda input_: input_.value.split('\t')).filter(
        lambda input_: len(input_) == 3).toDF().selectExpr("_1 as item1", "_2 as item2", "_3 as label")
    LabelDF = rawLabelDF
# read profile
# itemProfileDF = spark.read.text(itemProfilePath).rdd.repartition(100).filter(lambda x:x[0].split('|')[1] != 'pid').map(parseItemProfile).toDF().selectExpr("_1 as pid", "_2 as features")
#merge label and profile
def union_label_feature(labelDF, itemprofileDF):

    DF1 = labelDF.join(itemprofileDF, labelDF.item2 == itemprofileDF.pid, "left_outer").withColumnRenamed("value",
                                                                                                  "features2").drop(
        "pid").where(F.isnull('features2') == False)
    DF2 = DF1.join(itemprofileDF, DF1.item1 == itemprofileDF.pid, "left_outer").withColumnRenamed('value',
                                                                                                          'features1').drop(
        "pid").where(F.isnull('features1') == False)
    return DF2

if train:
    DF2_sample = union_label_feature(LabelDF_sample, itemProfileDF)
    DF2_sample = DF2_sample.withColumn('features1_features2', u_mergeTwoItemProfile(DF2_sample.features1, DF2_sample.features2))
    DF3_sample = DF2_sample.withColumn('merge', concat_ws(primary_delim, DF2_sample['ctr_label'], DF2_sample['uv_label'], DF2_sample['features1_features2']))
    DF2 = union_label_feature(LabelDF, itemProfileDF)
    DF2 = DF2.withColumn('features1_features2', u_mergeTwoItemProfile(DF2.features1, DF2.features2))
    DF3 = DF2.withColumn('merge', concat_ws(primary_delim, DF2['ctr_label'], DF2['uv_label'], DF2['features1_features2']))

    # trainDF, testDF = DF3.randomSplit([10.0, 1.0], 777)
    DF3_sample.select('merge').orderBy(rand()).write.text(trainDFOutPath)
    DF3.select('merge').orderBy(rand()).write.text(evalDFOutPath)
else:
    DF2 = union_label_feature(LabelDF, itemProfileDF)
    DF2 = DF2.withColumn('features1_features2', u_mergeTwoItemProfile(DF2.features1, DF2.features2))
    DF3 = DF2.withColumn('merge', concat_ws(primary_delim, DF2['item1'], DF2['item2'], DF2['features1_features2']))
    DF3.select('merge').write.text(predDFOutPath)