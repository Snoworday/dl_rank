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
import math

spark = SparkSession.builder.appName('tmp2').getOrCreate()
sc = spark.sparkContext

mode = 'gen_data'


def convertDate(date, delta):
    import datetime as dt
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date


parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default='{}'.format(str(datetime.date.today())))
_args = parser.parse_args()

itemid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/itemid/{}'.format(_args.date)
itemid_cross_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/idpair/{}'.format(_args.date)
cid_path = 's3://jiayun.spark.data/wangqi/wide_deep/Feature/cid/{}'.format(_args.date)
train_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/traindata/{}'.format(_args.date)
eval_out_path = 's3://jiayun.spark.data/wangqi/wide_deep/evaldata/{}'.format(_args.date)
item_profile_path = "s3://jiayun.spark.data/product_algorithm/product_statistics_features/item_profile_union/{}/".format(
    convertDate(_args.date, -3))
path = 's3://jiayun.spark.data/wangqi/wide_deep/I2ICTR/cid_imp_order_click/{}'.format(
    _args.date)  # /ctr_imp_order_click'

item_col = [328, 321]  # gmv, ctr
group_num = 3000

rate = 0.9
cid_rate = 0.7
pid_rate = 0.9
cross_rate = 0.4
max_row4one_cid = 2

data_stage2_pid_header = StructType([StructField(h, StringType(), True) for h in
                                     ['row', 'cid', 'pid', 'label', 'click_cross', 'order_cross', 'click_item']] + [
                                        StructField('idx_tag', IntegerType(), True)])
data_stage3_header = StructType(
    [StructField(h, StringType(), True) for h in ['cid', 'pid', 'label', 'click', 'click_cross', 'order_cross']])


def GenRecord(row):
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
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2] == 1],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 1],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 2])
               for target_idx in p_]
    n_click = [(cid,
                time_pid_tag[target_idx][1],
                0,
                [time_pid_tag[i][1] for i in range(target_idx) if time_pid_tag[i][2] == 1],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 1],
                [time_pid_tag[target_idx][1] + ',' + time_pid_tag[i][1] for i in range(target_idx) if
                 time_pid_tag[i][2] == 2])
               for target_idx in n_]
    sample_result = p_click + n_click
    return sample_result


def Flat_Table4cross(row):
    cid, pid, label, click, click_cross, order_cross = row[0]
    row_idx = row[1]
    click_order_cross = click_cross + order_cross
    seq_tag = len(click_cross) * [0] + len(order_cross) * [1]
    idx_tag = list(range(len(click_cross))) + list(range(len(order_cross)))
    out = [(row_idx, cid, pid, label, click, cross, idx, seq) for cross, idx, seq in
           zip(click_order_cross, idx_tag, seq_tag)]
    return out


def createCombiner4cross(v):
    cid, pid, label, click, idx_tag, seq_tag, idx = v
    if seq_tag == 0:
        return (cid, pid, label, click, [idx], [], [idx_tag], [])
    elif seq_tag == 1:
        return (cid, pid, label, click, [], [idx], [], [idx_tag])


def mergeValue4cross(c, v):
    c[4 + v[5]].append(v[6])
    c[6 + v[5]].append(v[4])
    return c


def mergeCombiners4cross(c1, c2):
    cid, pid, label, click, click_cross1, click_cross_idx1, order_cross1, order_cross_idx1 = c1
    _, _, _, _, click_cross2, click_cross_idx2, order_cross2, order_cross_idx2 = c2
    out = (cid, pid, label, click, click_cross1 + click_cross2, click_cross_idx1 + click_cross_idx2,
           order_cross1 + order_cross2, order_cross_idx1 + order_cross_idx2)
    return out


def Merge_Table4cross(row):
    row_num = row[0]
    cid, pid, label, click, click_cross, order_cross, click_cross_idx, order_cross_idx = row[1]
    try:
        click_cross = [click_cross[idx] for _, idx in sorted(zip(click_cross_idx, range(len(click_cross_idx))))]
    except:
        assert False, (click_cross_idx, click_cross)
    click_cross = str(click_cross)[1:-1]
    order_cross = [order_cross[idx] for _, idx in sorted(zip(order_cross_idx, range(len(order_cross_idx))))]
    order_cross = str(order_cross)[1:-1]
    out = (row_num, (cid, pid, label, click, click_cross, order_cross))
    return out


def Flat_Table4itemid(row):
    row_num = row[0]
    cid, pid, label, click, click_cross, order_cross = row[1]
    out = [(row_num, cid, pid, label, click_cross, order_cross, click_item, idx) for click_item, idx in
           zip(click, range(len(click)))]
    return out


def createCombiner4pid(v):
    cid, pid, label, click_cross, order_cross, idx_tag, idx = v
    out = (cid, pid, label, click_cross, order_cross, [idx_tag], [idx])
    return out


def mergeValue4pid(c, v):
    c[5].append(v[5])
    c[6].append(v[6])
    return c


def mergeCombiners4pid(c1, c2):
    cid, pid, label, click_cross, order_cross, idx_tag, idx = c1
    idx_tags = c1[5] + c2[5]
    idxs = c1[6] + c2[6]
    out = (cid, pid, label, click_cross, order_cross, idx_tags, idxs)
    return out


def Merge_Table4pid(row):
    row_num = row[0]
    cid, pid, label, click_cross, order_cross, idx_tag, click_idx = row[1]
    click_seq = [click_idx[idx] for _, idx in sorted(zip(idx_tag, range(len(idx_tag))))]
    click_seq = str(click_seq)[1:-1]
    out = (cid, pid, label, click_seq, click_cross, order_cross)
    return out

from pyspark.sql.window import Window
from pyspark.sql.functions import sum, desc

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


if mode == 'gen_data':
    ## itemid_path
    itemid = spark.read.text(itemid_path)
    new_itemid_row = spark.createDataFrame([[''], ['DEFAULT']])
    itemid = new_itemid_row.union(itemid).rdd.zipWithIndex().map(
        lambda x: (x[0][0], x[1] - 1))  # .collectAsMap()#.toDF()
    itemid = itemid.toDF(['key', 'idx'])
    ## cid_path
    cid = spark.read.text(cid_path)
    new_cid_row = spark.createDataFrame([[''], ['DEFAULT']])
    cid = new_cid_row.union(cid).rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]))  # .collectAsMap()#.toDF()
    cid = cid.toDF(['_id', 'idx'])
    ## itemid_cross_path
    itemid_cross = spark.read.text(itemid_cross_path)
    new_itemid_cross_row = spark.createDataFrame([[''], ['DEFAULT']])
    itemid_cross = new_itemid_cross_row.union(itemid_cross).rdd.zipWithIndex().map(
        lambda x: (x[0][0], x[1] - 1))  # .collectAsMap()#.toDF()
    itemid_cross = itemid_cross.toDF(['_id', 'idx'])
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
    item_profile_min_interv = tuple((_min, (_max - _min) / (group_num - 1)) for _min, _max in item_profile_min_max)
    item_profile_normal = item_profile.map(lambda x: tuple(
        [x[0]] + [math.ceil((i - min_interv[0]) / min_interv[1]) for i, min_interv in
                  zip(x[1:], item_profile_min_interv)]))
    item_profile = item_profile_normal.toDF(['key'] + [str(item) for item in item_col])

    ## read data
    data = sc.textFile(path)
    data_stage1 = data.repartition(1000).flatMap(GenRecord).zipWithIndex(). \
        flatMap(Flat_Table4cross). \
        toDF(['row', 'cid', 'pid', 'label', 'click', 'cross', 'idx_tag', 'seq_tag'])

    data_stage1_cross = data_stage1.join(itemid_cross, data_stage1.cross == itemid_cross._id, 'left').drop(
        'cross').drop('_id').na.fill(0) \
        .rdd.map(lambda x: (x[0], tuple(x[1:])))
    data_stage2 = data_stage1_cross.combineByKey(createCombiner=createCombiner4cross, mergeValue=mergeValue4cross,
                                                 mergeCombiners=mergeCombiners4cross).map(Merge_Table4cross)

    data_stage2_pid = data_stage2.flatMap(Flat_Table4itemid). \
        toDF(data_stage2_pid_header)
    data_stage2_pid = data_stage2_pid.join(itemid, data_stage2_pid.click_item == itemid.key, 'left').drop(
        'click_item').drop('key').na.fill(0). \
        rdd.map(lambda x: (x[0], tuple(x[1:])))
    data_stage3 = data_stage2_pid.combineByKey(createCombiner=createCombiner4pid, mergeValue=mergeValue4pid,
                                               mergeCombiners=mergeCombiners4pid). \
        map(Merge_Table4pid). \
        toDF(data_stage3_header)

    data_stage3_cid = data_stage3.join(cid, cid._id == data_stage3.cid, 'left'). \
        drop('_id').drop('cid').withColumnRenamed('idx', 'cid')
    #     assert False, data_stage3_cid.columns
    data_stage3_item_profile = data_stage3_cid.join(item_profile, data_stage3_cid.pid == item_profile.key, 'left'). \
        drop('key')
    #     AssertionError: (['pid', 'label', 'click', 'click_cross', 'order_cross', 'cid', '328', '321'], ['_id', 'idx'])
    data_stage4 = data_stage3_item_profile.join(itemid, data_stage3_item_profile.pid == itemid.key, 'left'). \
        drop('key').drop('pid').withColumnRenamed('idx', 'pid').na.fill(0)

    data_stage4_1 = data_stage4.withColumn('features',
                                           concat_ws('|', data_stage4.cid, data_stage4.pid, data_stage4.click,
                                                     data_stage4.click_cross, data_stage4.order_cross,
                                                     *[getattr(data_stage4, str(col)) for col in item_col]))
    data_stage4_2 = data_stage4_1.withColumn('merge', concat_ws('@', data_stage4_1.label, data_stage4_1.features))

    trainDF, testDF = data_stage4_2.randomSplit([rate, 1.0 - rate], 777)
    trainDF.select('merge').write.text(train_out_path)
    testDF.select('merge').write.text(eval_out_path)
else:
    data = sc.textFile(path)
    data_stage1 = data.repartition(10000).flatMap(GenRecord).flatMap(Fill_Split_Flat4count)  # .toDF().na.fill('')
    data_stage1.persist()
    count_all = data_stage1.count()
    cid_count = cid_rate * count_all
    pid_count = pid_rate * count_all
    cross_count = cross_rate * count_all
    # statistic high freq item
    cid = Filter_freq(data_stage1.map(lambda x: (x[0], 1)).filter(lambda x: x[0] != ''), cid_count)
    pid = Filter_freq(data_stage1.map(lambda x: (x[1], 1)).filter(lambda x: x[0] != ''), pid_count)
    cross = Filter_freq(data_stage1.map(lambda x: (x[2], 1)).filter(lambda x: x[0] != ''), cross_count)
    # save
    cid.write.text(cid_path)
    pid.write.text(itemid_path)
    cross.write.text(itemid_cross_path)













