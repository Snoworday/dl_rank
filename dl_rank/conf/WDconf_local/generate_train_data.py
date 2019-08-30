import pyspark
from pyspark import SQLContext
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

itemid_path = 's3://jiayun.spark.data/limengmeng/Deep&Wide/Feature/itemid'
itemid_cross_path = 's3://jiayun.spark.data/limengmeng/Deep&Wide/Feature/idpair'
cid_path = 's3://jiayun.spark.data/limengmeng/Deep&Wide/Feature/cid'
train_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/traindata'
eval_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/evaldata'


spark = SparkSession.builder.appName('tmp').getOrCreate()
sc = spark.sparkContext
spark.conf.set("spark.sql.crossJoin.enabled", "true")

click_cross_header = ['click_cross_{}'.format(i) for i in range(20)]
order_cross_header = ['order_cross_{}'.format(i) for i in range(20)]
click_header = ['click_{}'.format(i) for i in range(20)]
header_org = ['cid', 'trigger_id', 'label', 'click', 'order']
header = ['cid', 'trigger_id', 'label', 'click', 'order', 'cid_idx', 'trigger_id_idx', 'click_idx',
          'click_cross_idx', 'order_cross_idx']
# itemid_header = click_header
# itemid_cross_header = click_cross_header + order_cross_header
output_row = Row(*header)
header_schema_org = StructType([StructField(h, StringType(), True) for h in header_org])
header_schema = StructType([StructField(h, StringType(), True) for h in header])

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
    cid, imp, imptime, order, ordertime, click, clicktime = row[0].split('|')
    imp, order, click = imp.split(','), order.split(','), click.split(',')
    pid = [_ for _ in imp + order + click if _ != '']
    imptime, ordertime, clicktime = imptime.split(','), ordertime.split(','), clicktime.split(',')
    imptime = [] if imptime == [''] else imptime
    ordertime = [] if ordertime == [''] else ordertime
    clicktime = [] if clicktime == [''] else clicktime
    imp = [] if imp == [''] else imp
    order = [] if order == [''] else order
    click = [] if click == [''] else click
    time = [datetime.datetime.strptime(t.split('.')[0], "%Y-%m-%d %H:%M:%S") for t in imptime+ordertime+clicktime]
    tag = [0]*len(imptime) + [2]*len(ordertime) + [1]*len(clicktime)
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
    p_click = [(cid, time_pid_tag[target_idx][1], 1, [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],[time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==2])
               for target_idx in p_]
    n_click = [(cid, time_pid_tag[target_idx][1], 0, [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==1],[time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2]==2])
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

## data_path
path = 's3://jiayun.spark.data/wangqi/wide_deep/I2ICTR/cid_imp_order_click/2019-08-20'#/ctr_imp_order_click'
data = spark.read.text(path)
## item_path
itemid = spark.read.text(itemid_path)
new_itemid_row = spark.createDataFrame([[''], ['DEFAULT']])
itemid = new_itemid_row.union(itemid).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]-1)).collectAsMap()#.toDF()
b_itemid = sc.broadcast(itemid)
# itemid = itemid.withColumnRenamed('_1', 'pid').withColumnRenamed('_2', 'field_id')

itemid_cross = spark.read.text(itemid_cross_path)
new_itemid_cross_row = spark.createDataFrame([[''], ['DEFAULT']])
itemid_cross = new_itemid_cross_row.union(itemid_cross).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]-1)).collectAsMap()#.toDF()
b_itemid_cross = sc.broadcast(itemid_cross)
# itemid_cross = itemid_cross.withColumnRenamed('_1', 'pid').withColumnRenamed('_2', 'field_id')


cid = spark.read.text(cid_path)
new_cid_row = spark.createDataFrame([[''], ['DEFAULT']])
cid = new_cid_row.union(cid).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1])).collectAsMap()#.toDF()
b_cid = sc.broadcast(cid)
# cid = cid.withColumnRenamed('_1', 'pid').withColumnRenamed('_2', 'field_id')


data_stage1 = data.rdd.repartition(10000).flatMap(GenRecord).toDF(header_schema_org).na.fill('')
data_stage2 = data_stage1.rdd.map(lambda row: Gen_rows(row)).toDF(header_schema)

data_stage3 = data_stage2.withColumn('features', concat_ws('|', data_stage2['cid_idx'], data_stage2['trigger_id_idx'], data_stage2['click_idx'], data_stage2['click_cross_idx'], data_stage2['order_cross_idx']))
data_stage4 = data_stage3.select(['features', 'label']).withColumn('merge', concat_ws('@', data_stage3['label'], data_stage3['features']))

trainDF, testDF = data_stage4.randomSplit([9.0, 1.0], 777)
trainDF.select('merge').write.text(train_out_path)
testDF.select('merge').write.text(eval_out_path)



