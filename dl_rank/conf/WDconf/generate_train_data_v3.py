import pyspark
from pyspark import SparkConf
import datetime
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
import datetime
import math
from pyspark.sql.functions import explode
import pyspark.sql.functions as F
import operator
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, desc, collect_list
_args_date = '2019-09-03'
def convertDate(date, delta):
    import datetime as dt
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date
itemid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/itemid/{}'.format(_args_date)
itemid_cross_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/idpair/{}'.format(_args_date)
cid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/cid/{}'.format(_args_date)
train_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/traindata/{}'.format(_args_date)
eval_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/evaldata/{}'.format(_args_date)
item_profile_path = "s3://jiayun.spark.data/product_algorithm/product_statistics_features/item_profile_union/{}/".format(
    convertDate(_args_date, -3))
path = 's3://jiayun.spark.data/wangqi/wide_deep/I2ICTR/cid_imp_order_click/{}'.format(
    _args_date)  # /ctr_imp_order_click'

spark = SparkSession.builder.appName('tmp2').getOrCreate()
sc = spark.sparkContext
random.seed(777)

item_col = [328, 321]  # gmv, ctr
group_num = 3000

rate = 0.9
cid_rate = 0.8
pid_rate = 0.9
cross_rate = 0.5
seq_len = 20
max_row4one_cid = 7


data_stage3_header = StructType([
    StructField('cid', StringType(), True),
    StructField('pid', StringType(), True),
    StructField('label', StringType(), True),
    StructField('click', ArrayType(StringType()), True),
    StructField('click_cross', ArrayType(StringType()), True),
    StructField('order_cross', ArrayType(StringType()), True),
    StructField('row_num', IntegerType(), True),
])


combine_f = udf(lambda row, item_list: [(row, idx, item) for idx, item in enumerate(item_list)],
              ArrayType(StructType([StructField('row_num', IntegerType(), True),
                                    StructField('idx', IntegerType(), True),
                                    StructField('item', StringType(), True)])))
sort_f = udf(lambda x: [ item[1] for item in sorted(x, key=operator.itemgetter(0))])

w = Window.partitionBy('row').orderBy('idx')

def GenRecord(row):
    '''
    cid, pid, label, pid_history, pid_cross_history(click), pid_cross_history(order)
    :param row: a row represent one User(cid) which combine all fields split by '|'
    :return: rdd(multi field), multi elems in each field(str) split by ','
    '''
    try:
        cid, imp, imptime, order, ordertime, click, clicktime = row.split('|')
    except:
        assert False, row
    imp, click, order = imp.split(','), click.split(','), order.split(',')
    pid = [_ for _ in imp + click + order if _ != '']
    imptime, clicktime, ordertime = imptime.split(','), clicktime.split(','), ordertime.split(',')
    imptime = [] if imptime == [''] else imptime
    clicktime = [] if clicktime == [''] else clicktime
    ordertime = [] if ordertime == [''] else ordertime
    # imp = [] if imp == [''] else imp
    # order = [] if order == [''] else order
    # click = [] if click == [''] else click
    time = [datetime.datetime.strptime(t.split('.')[0], "%Y-%m-%d %H:%M:%S") for t in imptime + clicktime + ordertime]
    tag = [0] * len(imptime) + [1] * len(clicktime) + [2] * len(ordertime)
    time_pid_tag = list(zip(time, pid, tag))

    time_pid_tag.sort()
    p_ = [i for i, x in enumerate(time_pid_tag) if x[2] == 1]
    n_ = [i for i, x in enumerate(time_pid_tag) if x[2] == 0]
    random.shuffle(p_)
    random.shuffle(n_)
    num_ = min(len(p_), len(n_), max_row4one_cid)
    #     assert False, time
    assert len(time) == len(pid) and len(pid) == len(tag), 'emm'
    p_ = p_[:num_]
    n_ = n_[:num_]
    p_click = [(cid,
                time_pid_tag[target_idx][1],
                1,
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2] == 1][-seq_len:],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 1][-seq_len:],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 2][-seq_len:])
               for target_idx in p_]
    n_click = [(cid,
                time_pid_tag[target_idx][1],
                0,
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2] == 1][-seq_len:],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 1][-seq_len:],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 2][-seq_len:])
               for target_idx in n_]
    sample_result = p_click + n_click
    return sample_result

def Fill_Split_Flat4count(row):
    cid, pid, label, click_seq, click_cross_seq, order_cross_seq = row
    pid = [pid, *click_seq]
    cross = [*click_cross_seq, *order_cross_seq]
    max_len = max(1, len(cross), len(pid))
    pid = pid + (max_len - len(pid)) * ['']
    cross = cross + (max_len - len(cross)) * ['']
    cid = max_len * [cid]
    out = list(zip(cid, pid, cross))
    return out

def Filter_freq(rdd, d_count):
    w = Window.orderBy(desc('_count')).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    rdd_out = rdd. \
        filter(lambda x: x[0] != ''). \
        reduceByKey(lambda x, y: x + y)
    df = rdd_out.toDF(['_id', '_count'])
    df = df.withColumn('acc', sum(df._count).over(w))
    df = df.filter(df['acc'] < d_count).sort(desc('_count')).select('_id')
        # sortBy(lambda x: x[1], ascending=False). \
        # flatMap(lambda x: [x[0]] * x[1]).zipWithIndex(). \
        # filter(lambda x: x[1] < d_count). \
        # reduceByKey(lambda x, y: x). \
        # map(lambda x: x[0])
        # # map(lambda x: x[0]). \
        # # distinct()
    return df

def embbeding_list_field(df, label_df, df_name):
    out = df.withColumn('new', combine_f('row_num', df_name))\
        .withColumn('new', explode('new'))\
        .select(F.col('new.row_num').alias('row_num'), F.col('new.idx').alias('idx'), F.col('new.item').alias('item'))\
        .alias('tmp')\
        .join(label_df.alias('b'), F.col('tmp.item')==F.col('b.key'), 'left').na.fill(0)\
        .drop('key').drop('item')\
        .groupBy('row_num').agg(collect_list(F.struct('idx', 'value')).alias('idx_value'))\
        .withColumn(df_name, sort_f('idx_value')).drop('idx_value')
    return out


#----------------------Generate embedding label-------------------------------------------------------------------------
data = sc.textFile(path)
data_stage1 = data.repartition(10000).flatMap(GenRecord).flatMap(Fill_Split_Flat4count)  # .toDF().na.fill('')
data_stage1.persist()
count_all = data_stage1.count()
cid_count = cid_rate * count_all
pid_count = pid_rate * count_all
cross_count = cross_rate * count_all
pid = Filter_freq(data_stage1.map(lambda x: (x[1], 1)).filter(lambda x: x[0] != ''), pid_count)
cid = Filter_freq(data_stage1.map(lambda x: (x[0], 1)).filter(lambda x: x[0] != ''), cid_count)
cross = Filter_freq(data_stage1.map(lambda x: (x[2], 1)).filter(lambda x: x[0] != ''), cross_count)


#----------------------Generate train/eval data-------------------------------------------------------------------------

# Reshape embedding labels
itemid = pid # spark.read.text(itemid_path)
new_itemid_row = spark.createDataFrame([['DEFAULT']])
itemid_append = new_itemid_row.union(itemid)
pid_emb = itemid_append.rdd.zipWithIndex().map(lambda x: (x[0][0], x[1])).toDF(['key', 'value'])

## cid_path
cid = cid # spark.read.text(cid_path)
new_cid_row = spark.createDataFrame([['DEFAULT']])
cid_append = new_cid_row.union(cid)
cid_emb = cid_append.rdd.zipWithIndex().map(lambda x: (x[0][0], x[1])).toDF(['key', 'value'])

## itemid_cross_path
itemid_cross = cross # spark.read.text(itemid_cross_path)
new_itemid_cross_row = spark.createDataFrame([['DEFAULT']])
itemid_cross_append = new_itemid_cross_row.union(itemid_cross)
cross_emb = itemid_cross_append.rdd.zipWithIndex().map(lambda x: (x[0][0], x[1])).toDF(['key', 'value'])

## item_profile_path
item_profile = sc.textFile(item_profile_path).zipWithIndex().filter(lambda x: x[1] != 0).map(lambda x: x[0])
item_profile = item_profile.map(
    lambda x: tuple([x.split('|')[1]] + [float(x.split('|')[col]) if x.split('|')[col] != '' else 0 for col in
                                         item_col]))
item_profile_min_max = item_profile.aggregate(tuple([(math.inf, -math.inf)] * len(item_col)),
                                              seqOp=lambda c, v: tuple(
                                                  [(min(c[i][0], v[i + 1]), max(c[i][1], v[i + 1])) for i in
                                                   range(len(v) - 1)]),
                                              combOp=lambda c1, c2: tuple(
                                                  [(min(c_1[0], c_2[0]), max(c_1[1], c_2[1])) for c_1, c_2 in
                                                   zip(c1, c2)]))
item_profile_min_max = ((0, 70000000.0), (0, 1800.0))
item_profile_min_interv = tuple((_min, (_max - _min) / (group_num - 1)) for _min, _max in item_profile_min_max)
item_profile_normal = item_profile.map(lambda x: tuple(
    [x[0]] + [math.ceil((i - min_interv[0]) / min_interv[1]) for i, min_interv in
              zip(x[1:], item_profile_min_interv)]))
item_profile = item_profile_normal.toDF(['key'] + [str(item) for item in item_col])

# save label with Default
cid_append.write.text(cid_path)
itemid_append.write.text(itemid_path)
itemid_cross_append.write.text(itemid_cross_path)


# start embedding
data_sampled_df = data.flatMap(GenRecord).zipWithIndex().map(lambda x: (*x[0], x[1])).toDF(data_stage3_header).persist()
data_cid = data_sampled_df.select('cid', 'row_num')
data_pid = data_sampled_df.select('pid', 'row_num')
data_click = data_sampled_df.select('click', 'row_num')
data_click_cross = data_sampled_df.select('click_cross', 'row_num')
data_order_cross = data_sampled_df.select('order_cross', 'row_num')
data_label = data_sampled_df.select('label', 'row_num')

# embeding for each field
data_cid_emb = data_cid.join(cid_emb, data_cid.cid==cid_emb.key, 'left').na.fill(0).drop('key').drop('cid').withColumnRenamed('value', 'cid')

data_pid_profile = data_pid.join(item_profile, data_pid.pid==item_profile.key).drop('key')
data_pid_profile_emb = data_pid_profile.join(pid_emb, data_pid_profile.pid==pid_emb.key, 'left').na.fill(0).drop('key').drop('pid').withColumnRenamed('value', 'pid')

data_click_emb = embbeding_list_field(data_click, pid_emb, 'click')

data_click_cross_emb = embbeding_list_field(data_click_cross, cross_emb, 'click_cross')
data_order_cross_emb = embbeding_list_field(data_order_cross, cross_emb, 'order_cross')

# merge according to row number
result = data_label.join(data_cid_emb, data_label.row_num==data_cid_emb.row_num, 'inner').drop(data_label['row_num'])
result = result.join(data_pid_profile_emb, result.row_num==data_pid_profile_emb.row_num, 'inner').drop(result['row_num'])
result = result.join(data_click_emb, result.row_num==data_click_emb.row_num, 'inner').drop(result['row_num'])
result = result.join(data_click_cross_emb, result.row_num==data_click_cross_emb.row_num, 'inner').drop(result['row_num'])
result = result.join(data_order_cross_emb, result.row_num==data_order_cross_emb.row_num, 'inner').drop(result['row_num'])

data_stage4_1 = result.withColumn('features',
                                       concat_ws('|', result.cid, result.pid, result.click,
                                                 result.click_cross, result.order_cross,
                                                 *[getattr(result, str(col)) for col in item_col]))
data_stage4_2 = data_stage4_1.withColumn('merge', concat_ws('@', data_stage4_1.label, data_stage4_1.features))

trainDF, testDF = data_stage4_2.orderBy(rand()).randomSplit([rate, 1.0 - rate], 777)
trainDF.select('merge').rdd.map(lambda x: '|'.join([item[1:-1] if idx in [2,3,4] else item for idx, item in enumerate(x[0].split('|'))])).saveAsTextFile(train_out_path)
testDF.select('merge').rdd.map(lambda x: '|'.join([item[1:-1] if idx in [2,3,4] else item for idx, item in enumerate(x[0].split('|'))])).saveAsTextFile(eval_out_path)
