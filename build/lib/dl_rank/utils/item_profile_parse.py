from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, concat_ws
import pyspark.sql.functions as f
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import logging
import argparse
# from util.data_info import item_profile_head_4_1 as head
DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument("--date", help="number of records per batch", type=str)
parser.add_argument("--mode", help="[train/infer]", type=str)
_args = parser.parse_args()
spark = SparkSession.builder.appName('merge_label_itemProfile').getOrCreate()

train = _args.mode == 'train'

head = 'log_date|pid|pno|list_price|c_platform_price|discount|is_sale|create_date|sku_num|catid1|catid2|catid3|cat1_price|cat2_price|cat3_price|imp_uv_1d|click_uv_1d|ctr_uv_1d|acr_uv_1d|wr_uv_1d|imp_uv_7d|click_uv_7d|ctr_uv_7d|acr_uv_7d|wr_uv_7d|imp_uv_15d|click_uv_15d|ctr_uv_15d|acr_uv_15d|wr_uv_15d|imp_uv_30d|click_uv_30d|ctr_uv_30d|acr_uv_30d|wr_uv_30d|imp_uv_60d|click_uv_60d|ctr_uv_60d|acr_uv_60d|wr_uv_60d|imp_uv_90d|click_uv_90d|ctr_uv_90d|acr_uv_90d|wr_uv_90d|comment_cnt_1d|score_description_1d|score_quality_1d|score_size_1d|score_1d|good_score_rate_1d|imp_1d|click_1d|add_1d|add_uv_1d|wishlist_1d|wishlist_uv_1d|ctr_1d|acr_1d|wr_1d|sales_1d|in_sales_1d|orders_1d|price_1d|gmv_1d|buyers_1d|in_buyers_1d|buyer_male_1d|buyer_female_1d|buyer_neutral_1d|male_rate_1d|female_rate_1d|neutral_rate_1d|confirm_sales_1d|in_confirm_sales_1d|confirm_orders_1d|confirm_price_1d|confirm_gmv_1d|confirm_buyers_1d|in_confirm_buyers_1d|confirm_buyer_male_1d|confirm_buyer_female_1d|confirm_buyer_neutral_1d|confirm_male_rate_1d|confirm_female_rate_1d|confirm_neutral_rate_1d|refund_1d|refund_rate_1d|repurchase_rate_1d|comment_cnt_7d|score_description_7d|score_quality_7d|score_size_7d|score_7d|good_score_rate_7d|imp_7d|click_7d|add_7d|add_uv_7d|wishlist_7d|wishlist_uv_7d|ctr_7d|acr_7d|wr_7d|sales_7d|in_sales_7d|orders_7d|price_7d|gmv_7d|buyers_7d|in_buyers_7d|buyer_male_7d|buyer_female_7d|buyer_neutral_7d|male_rate_7d|female_rate_7d|neutral_rate_7d|confirm_sales_7d|in_confirm_sales_7d|confirm_orders_7d|confirm_price_7d|confirm_gmv_7d|confirm_buyers_7d|in_confirm_buyers_7d|confirm_buyer_male_7d|confirm_buyer_female_7d|confirm_buyer_neutral_7d|confirm_male_rate_7d|confirm_female_rate_7d|confirm_neutral_rate_7d|refund_7d|refund_rate_7d|repurchase_rate_7d|comment_cnt_15d|score_description_15d|score_quality_15d|score_size_15d|score_15d|good_score_rate_15d|imp_15d|click_15d|add_15d|add_uv_15d|wishlist_15d|wishlist_uv_15d|ctr_15d|acr_15d|wr_15d|sales_15d|in_sales_15d|orders_15d|price_15d|gmv_15d|buyers_15d|in_buyers_15d|buyer_male_15d|buyer_female_15d|buyer_neutral_15d|male_rate_15d|female_rate_15d|neutral_rate_15d|confirm_sales_15d|in_confirm_sales_15d|confirm_orders_15d|confirm_price_15d|confirm_gmv_15d|confirm_buyers_15d|in_confirm_buyers_15d|confirm_buyer_male_15d|confirm_buyer_female_15d|confirm_buyer_neutral_15d|confirm_male_rate_15d|confirm_female_rate_15d|confirm_neutral_rate_15d|refund_15d|refund_rate_15d|repurchase_rate_15d|comment_cnt_30d|score_description_30d|score_quality_30d|score_size_30d|score_30d|good_score_rate_30d|imp_30d|click_30d|add_30d|add_uv_30d|wishlist_30d|wishlist_uv_30d|ctr_30d|acr_30d|wr_30d|sales_30d|in_sales_30d|orders_30d|price_30d|gmv_30d|buyers_30d|in_buyers_30d|buyer_male_30d|buyer_female_30d|buyer_neutral_30d|male_rate_30d|female_rate_30d|neutral_rate_30d|confirm_sales_30d|in_confirm_sales_30d|confirm_orders_30d|confirm_price_30d|confirm_gmv_30d|confirm_buyers_30d|in_confirm_buyers_30d|confirm_buyer_male_30d|confirm_buyer_female_30d|confirm_buyer_neutral_30d|confirm_male_rate_30d|confirm_female_rate_30d|confirm_neutral_rate_30d|refund_30d|refund_rate_30d|repurchase_rate_30d|comment_cnt_60d|score_description_60d|score_quality_60d|score_size_60d|score_60d|good_score_rate_60d|imp_60d|click_60d|add_60d|add_uv_60d|wishlist_60d|wishlist_uv_60d|ctr_60d|acr_60d|wr_60d|sales_60d|in_sales_60d|orders_60d|price_60d|gmv_60d|buyers_60d|in_buyers_60d|buyer_male_60d|buyer_female_60d|buyer_neutral_60d|male_rate_60d|female_rate_60d|neutral_rate_60d|confirm_sales_60d|in_confirm_sales_60d|confirm_orders_60d|confirm_price_60d|confirm_gmv_60d|confirm_buyers_60d|in_confirm_buyers_60d|confirm_buyer_male_60d|confirm_buyer_female_60d|confirm_buyer_neutral_60d|confirm_male_rate_60d|confirm_female_rate_60d|confirm_neutral_rate_60d|refund_60d|refund_rate_60d|repurchase_rate_60d|comment_cnt_90d|score_description_90d|score_quality_90d|score_size_90d|score_90d|good_score_rate_90d|imp_90d|click_90d|add_90d|add_uv_90d|wishlist_90d|wishlist_uv_90d|ctr_90d|acr_90d|wr_90d|sales_90d|in_sales_90d|orders_90d|price_90d|gmv_90d|buyers_90d|in_buyers_90d|buyer_male_90d|buyer_female_90d|buyer_neutral_90d|male_rate_90d|female_rate_90d|neutral_rate_90d|confirm_sales_90d|in_confirm_sales_90d|confirm_orders_90d|confirm_price_90d|confirm_gmv_90d|confirm_buyers_90d|in_confirm_buyers_90d|confirm_buyer_male_90d|confirm_buyer_female_90d|confirm_buyer_neutral_90d|confirm_male_rate_90d|confirm_female_rate_90d|confirm_neutral_rate_90d|refund_90d|refund_rate_90d|repurchase_rate_90d|comment_cnt|score_description|score_quality|score_size|score|good_score_rate|imp|click|add|add_uv|wishlist|wishlist_uv|ctr|acr|wr|sales|in_sales|orders|price|gmv|buyers|in_buyers|buyer_male|buyer_female|buyer_neutral|male_rate|female_rate|neutral_rate|confirm_sales|in_confirm_sales|confirm_orders|confirm_price|confirm_gmv|confirm_buyers|in_confirm_buyers|confirm_buyer_male|confirm_buyer_female|confirm_buyer_neutral|confirm_male_rate|confirm_female_rate|confirm_neutral_rate|refund|refund_rate|repurchase_rate|page_imp_1d|page_imp_uv_1d|page_imp_7d|page_imp_uv_7d|page_imp_15d|page_imp_uv_15d|page_imp_30d|page_imp_uv_30d|page_imp_60d|page_imp_uv_60d|page_imp_90d|page_imp_uv_90d|page_imp|page_imp_uv|season|gender|illegal_tags|shipping_sales_1d|shipping_orders_1d|shipping_sales_7d|shipping_orders_7d|shipping_sales_15d|shipping_orders_15d|shipping_sales_30d|shipping_orders_30d|shipping_sales_60d|shipping_orders_60d|shipping_sales_90d|shipping_orders_90d|shipping_sales|shipping_orders|confirm_price_unit_india_var_1d|confirm_price_unit_india_var_7d|confirm_price_unit_india_var_15d|confirm_price_unit_india_var_30d|confirm_price_unit_india_var_60d|confirm_price_unit_india_var_90d|confirm_price_unit_india_var|confirm_price_unit_india_avg_1d|confirm_price_unit_india_avg_7d|confirm_price_unit_india_avg_15d|confirm_price_unit_india_avg_30d|confirm_price_unit_india_avg_60d|confirm_price_unit_india_avg_90d|confirm_price_unit_india_avg'

ratioSample = 0.1#_
headlist = head.split('|')
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
        labelPath = 's3://jiayun.spark.data/wangqi/ProductDetail/I2IRank/rawlabel/{}'.format(_args.date)
        trainDFOutPath = "s3://jiayun.spark.data/wangqi/I2IRankData/TextFile/TrainData/{}".format(_args.date)
        evalDFOutPath = "s3://jiayun.spark.data/wangqi/I2IRankData/TextFile/EvalData/{}".format(_args.date)
    else:
        labelPath = 's3://jiayun.spark.data/songfeng/product/swing_time1/headHold_5-rankHold_400-dayHold1_72-dayHold2_360-tailHold_700/cate_sort/csv/'
        # labelPath = 's3://jiayun.spark.data/wangqi//I2IRankData/TextFile/LabelData/'
        predDFOutPath = 's3://jiayun.spark.data/wangqi/I2IRankData/TextFile/PredData/{}'.format(_args.date)
    itemProfilePath = "s3://jiayun.spark.data/product_algorithm/product_statistics_features/item_profile_union/{}/".format(convertDate(_args.date, -1))

colFactory = udf(lambda x: x.split('|')[1])
def checklabel(X):
    item1, item2, label = X
    item1 = item1.replace(" ",'')
    item2 = item2.replace(" ",'')
    label = label.replace(" ",'')
    if item1 == "" or item1.replace(" ", '') == "":
        return False
    if item2 == "" or item2.replace(" ", '') == "":
        return False
    if label == "" or label.replace(" ", '') == "":
        return False
    return True

def parseItemProfile(X):
    rawdata = X[0].split('|')
    # rawdata = dict(zip(headlist, rawdata))
    return (rawdata[1], str(X))

#read profile
itemProfileDF = spark.read.text(itemProfilePath).rdd.repartition(1000).filter(lambda x:x[0].split('|')[1] != 'pid').toDF()
itemProfileDF = itemProfileDF.withColumn('pid', colFactory(itemProfileDF.value))
#read label
if train:
    rawLabelDF = spark.read.text(labelPath).rdd.repartition(5000).map(lambda input_: input_.value.split('|')).filter(
        lambda input_: len(input_) == 3).filter(checklabel).toDF().selectExpr("_1 as item1", "_2 as item2",
                                                                              "_3 as label")
    recordCount = rawLabelDF.cache().count()
    testCount = recordCount//10
    trainCount = recordCount - testCount
    # read label
    rawLabelPositiveDF = rawLabelDF.filter(rawLabelDF[2] == 1)
    rawLabelNegativeDF = rawLabelDF.filter(rawLabelDF[2] == 0)
    PCount = rawLabelPositiveDF.count()
    NCount = rawLabelNegativeDF.count()
    ratio = PCount/NCount
    assert ratio < 1, 'ratio?:{}'.format(ratio)
    rawLabelNegativeDF_sample = rawLabelNegativeDF.sample(withReplacement=False, fraction=ratio)
    LabelDF_sample = rawLabelPositiveDF.union(rawLabelNegativeDF_sample)
    LabelDF = rawLabelPositiveDF.union(rawLabelNegativeDF)
else:
    rawLabelDF = spark.read.text(labelPath).rdd.repartition(1000).map(
        lambda input_: input_.value.split('\t')).filter(
        lambda input_: len(input_) == 3).filter(checklabel).toDF().selectExpr("_1 as item1", "_2 as item2",
                                                                              "_3 as label")
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
    DF3_sample = DF2_sample.withColumn('merge', concat_ws('@', DF2_sample['label'], DF2_sample['features1'], DF2_sample['features2']))
    DF2 = union_label_feature(LabelDF, itemProfileDF)
    DF3 = DF2.withColumn('merge', concat_ws('@', DF2['label'], DF2['features1'], DF2['features2']))

    # trainDF, testDF = DF3.randomSplit([10.0, 1.0], 777)
    DF3_sample.select('merge').write.text(trainDFOutPath)
    DF3.select('merge').write.text(evalDFOutPath)
else:
    DF2 = union_label_feature(LabelDF, itemProfileDF)
    DF3 = DF2.withColumn('merge', concat_ws('@', DF2['item1'], DF2['item2'], DF2['features1'], DF2['features2']))
    DF3.select('merge').write.text(predDFOutPath)