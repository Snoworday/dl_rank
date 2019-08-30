import pyspark
from pyspark import SparkConf
import datetime
import random
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from pyspark.sql.functions import concat, concat_ws, udf, shuffle, hash, broadcast
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import col, when
from pyspark.sql import Row
import argparse
import datetime

conf = SparkConf()
conf.set('spark.executor.instantces', '20')
conf.set('spark.executor.cores', '4')


mode = 'gen_voca'

def convertDate(date, delta):
    import datetime as dt
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default='{}'.format(str(datetime.date.today())))
_args = parser.parse_args()

itemid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/itemid'
itemid_cross_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/idpair'
cid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/cid'
train_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/traindata'
eval_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/evaldata'
item_profile_path = "s3://jiayun.spark.data/product_algorithm/product_statistics_features/item_profile_union/{}/".format(convertDate(_args.date, -1))
path = 's3://jiayun.spark.data/wangqi/wide_deep/I2ICTR/cid_imp_order_click/{}'.format(_args.date)  # /ctr_imp_order_click'

item_col = [328, 321]   #gmv, ctr
group_num = 3000
rate = 0.9

def Fill_Split(row):
    cid, pid, label, click_seq, order_seq = row
    click_cross_seq = [pid+','+click_id for click_id in click_seq]
    order_cross_seq = [pid+','+order_id for order_id in order_seq]
    click_seq += [None]*(20 - len(click_seq))
    order_seq += [None]*(20 - len(order_seq))
    click_cross_seq += [None]*(20 - len(click_cross_seq))
    order_cross_seq += [None]*(20 - len(order_cross_seq))
    out = (cid, pid, label, *click_cross_seq, *order_cross_seq, *click_seq)
    return out

def GenRecord(row):
    cid, imp, imptime, order, ordertime, click, clicktime = row.split('|')
    imp, click, order = imp.split(','), click.split(','), order.split(',')
    pid = [_ for _ in imp + click + order if _ != '']
    imptime, clicktime, ordertime = imptime.split(','), clicktime.split(','), ordertime.split(',')
    imptime = [] if imptime == [''] else imptime
    clicktime = [] if clicktime == [''] else clicktime
    ordertime = [] if ordertime == [''] else ordertime
    # imp = [] if imp == [''] else imp
    # order = [] if order == [''] else order
    # click = [] if click == [''] else click
    time = [datetime.datetime.strptime(t.split('.')[0], "%Y-%m-%d %H:%M:%S") for t in imptime+clicktime+ordertime]
    tag = [0]*len(imptime) + [1]*len(clicktime) + [2]*len(ordertime)
    time_pid_tag = list(zip(time, pid, tag))
    time_pid_tag.sort()
    p_ = [i for i, x in enumerate(time_pid_tag) if x[2]==1]
    n_ = [i for i, x in enumerate(time_pid_tag) if x[2]==0]
    random.shuffle(p_)
    random.shuffle(n_)
    num_ = min(len(p_), len(n_))
#     assert False, time
    assert len(time)==len(pid) and len(pid)==len(tag), 'emm'
    p_ = p_[:num_]
    n_ = n_[:num_+1]
    p_click = [(cid,
                time_pid_tag[target_idx][1],
                1,
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],
                [time_pid_tag[target_idx][1]+','+time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],
                [time_pid_tag[target_idx][1]+','+time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==2])
               for target_idx in p_]
    n_click = [(cid,
                time_pid_tag[target_idx][1],
                0,
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],
                [time_pid_tag[target_idx][1]+','+time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],
                [time_pid_tag[target_idx][1]+','+time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==2])
           for target_idx in n_]
    sample_result = p_click+n_click
    return sample_result

# def _Random_replace_null(x):
#     if x=='':
#         return str(random.randint(-1000, -1))
#     else:
#         return x
# Random_replace_null = udf(_Random_replace_null, T.StringType())

def Gen_rows(x):
    x_cid = x.cid
    x_trigger_id = x.trigger_id
    x_click = [i for i in x.click[1:-1].split(',') if i != '']
    x_order = x.order[1:-1].split(',')
    x_click_cross = [x_trigger_id+','+c for c in x_click]
    x_order_cross = [x_trigger_id+','+c for c in x_order if c != '']

    x_cid = b_cid.value.get(x_cid, 0)
    x_trigger_id = b_itemid.value.get(x_trigger_id, 0)
    x_click = [b_itemid.value.get(i, 0) for i in x_click]
    x_click_cross = [b_itemid_cross.value.get(i, 0) for i in x_click_cross]
    x_order_cross = [b_itemid_cross.value.get(i, 0) for i in x_order_cross]
    x_cid = (str(x_cid),)
    x_trigger_id = (str(x_trigger_id),)
    x_click = (str(x_click)[1:-1],)
    x_click_cross = (str(x_click_cross)[1:-1],)
    x_order_cross = (str(x_order_cross)[1:-1],)
    out = output_row(*x+x_cid+x_trigger_id+x_click+x_click_cross+x_order_cross)
    return out

def Fill_Split_Flat4count(row):
    cid, pid, label, click_seq, order_seq = row
    click_cross_seq = [pid+','+click_id for click_id in click_seq]
    order_cross_seq = [pid+','+order_id for order_id in order_seq]
    pid = [pid, *click_seq]
    cross = [*click_cross_seq, *order_cross_seq]
    max_len = max(1, len(cross), len(pid))
    pid = pid + (max_len-len(pid))*[None]
    cross = cross + (max_len-len(cross))*[None]
    cid = max_len * [cid]
    out = list(zip(cid, pid, cross))
    return out

def Filter_freq(rdd, d_count):
    rdd_out = rdd.\
        filter(lambda x: x[0]!='').\
        reduceByKey(lambda x, y: x+y).\
        sortBy(lambda x: x[1], ascending=False).\
        flatMap(lambda x: [x[0]]*x[1]).zipWithIndex().\
        filter(lambda x: x[1]< d_count).\
        map(lambda x: x[0]).\
        distinct()
        # reduceByKey(lambda x, y: x).\
        # map(lambda x: x[0])
    return rdd_out

import math
spark = SparkSession.builder.appName('tmp').getOrCreate()
sc = spark.sparkContext
if mode == 'gen_data':
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    click_cross_header = ['click_cross_{}'.format(i) for i in range(20)]
    order_cross_header = ['order_cross_{}'.format(i) for i in range(20)]
    click_header = ['click_{}'.format(i) for i in range(20)]
    header_org = ['cid', 'trigger_id', 'label', 'click', 'order']
    header = ['cid', 'trigger_id', 'label', 'click', 'order'] + item_col + ['cid_idx', 'trigger_id_idx', 'click_idx',
              'click_cross_idx', 'order_cross_idx']
    # itemid_header = click_header
    # itemid_cross_header = click_cross_header + order_cross_header
    output_row = Row(*header)
    header_schema_org = StructType([StructField(h, StringType(), True) for h in header_org])
    header_schema = StructType([StructField(h, StringType(), True) for h in header])
    ## data_path
    data = sc.textFile(path)
    ## item_path
    itemid = spark.read.text(itemid_path)
    new_itemid_row = spark.createDataFrame([[''], ['DEFAULT']])
    itemid = new_itemid_row.union(itemid).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]-1)).collectAsMap()#.toDF()
    b_itemid = sc.broadcast(itemid)
    ## cross_path
    itemid_cross = spark.read.text(itemid_cross_path)
    new_itemid_cross_row = spark.createDataFrame([[''], ['DEFAULT']])
    itemid_cross = new_itemid_cross_row.union(itemid_cross).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]-1)).collectAsMap()#.toDF()
    b_itemid_cross = sc.broadcast(itemid_cross)
    ## cid_path
    cid = spark.read.text(cid_path)
    new_cid_row = spark.createDataFrame([[''], ['DEFAULT']])
    cid = new_cid_row.union(cid).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1])).collectAsMap()#.toDF()
    b_cid = sc.broadcast(cid)
    ## item_profile_path
    item_profile = sc.textFile(item_profile_path).zipWithIndex().filter(lambda x: x[1] != 0).map(lambda x: x[0])
    item_profile = item_profile.map(lambda x: tuple([float(x.split('|')[col]) if x.split('|')[col]!='' else 0 for col in [1]+item_col])).persist()
    item_profile_min_max = item_profile.aggregate([(math.inf, -math.inf)]*len(item_col),
                                                                seqOp=lambda c, v: tuple([(min(c[i][0], v[i]), max(c[i][1], v[i])) for i in range(1, len(v))]),
                                                                combOp=lambda c1, c2: tuple([(min(c_1[0], c_2[0]), max(c_1[1], c_2[1])) for c_1, c_2 in zip(c1, c2)]))
    item_profile_min_interv = tuple((_min, (_max-_min)/(group_num-1)) for _min, _max in item_profile_min_max)
    item_profile_normal = item_profile.map(lambda x: tuple([x[0]] + [math.ceil((i-min_interv[0])/min_interv[1]) for i, min_interv in zip(x[1:], item_profile_min_interv)]))
    item_profile_normal_df = item_profile_normal.toDF('pid', *item_col)

    data_stage1 = data.repartition(10000).flatMap(GenRecord).toDF(header_schema_org).na.fill('')
    data_stage1_item = data_stage1.join(item_profile_normal, data_stage1.trigger_id==item_profile_normal.pid, 'leftjoin').na.fill(0)

    data_stage2 = data_stage1_item.rdd.map(lambda row: Gen_rows(row)).toDF(header_schema)

    data_stage3 = data_stage2.withColumn('features', concat_ws('|', data_stage2['cid_idx'], data_stage2['trigger_id_idx'], data_stage2['click_idx'], data_stage2['click_cross_idx'], data_stage2['order_cross_idx'],
                                                               *[data_stage2[col] for col in item_col]))
    data_stage4 = data_stage3.select(['features', 'label']).withColumn('merge', concat_ws('@', data_stage3['label'], data_stage3['features']))

    trainDF, testDF = data_stage4.randomSplit([9.0, 1.0], 777)
    trainDF.select('merge').write.text(train_out_path)
    testDF.select('merge').write.text(eval_out_path)
elif mode == 'gen_voca':
    data = spark.read.text(path)
    data_stage1 = data.rdd.repartition(10000).flatMap(GenRecord).flatMap(Fill_Split_Flat4count)  # .toDF().na.fill('')
    data_stage1.persist()
    count_all = data_stage1.count()
    d_count = rate * count_all
    # statistic high freq item
    cid = Filter_freq(data_stage1.map(lambda x: (x[0], 1)), d_count)
    pid = Filter_freq(data_stage1.map(lambda x: (x[1], 1)), d_count)
    cross = Filter_freq(data_stage1.map(lambda x: (x[2], 1)), d_count)
    # save
    cid.saveAsTextFile(cid_path)
    pid.saveAsTextFile(itemid_path)
    cross.saveAsTextFile(itemid_cross_path)

# 328:gmv, 321:ctr

