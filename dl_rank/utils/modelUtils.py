#config=utf-8
import tensorflow as tf
from tensorflow import gfile
import os
import shutil
import numpy as np
import time
import argparse
import sys
import logging
# get TF logger
import subprocess
import re
import datetime as dt
from functools import reduce
from collections import defaultdict


def open_log(path, mode):
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path, 'dl_rank_'+mode+'.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    # os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'

def convertDate(date, delta):
    if date == '':
        return date
    date = date.split(':')[-1]
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date

def _resc(dict):
    if tf.gfile.IsDirectory(dict):
        out = []
        for fp, dirs, files in tf.gfile.Walk(dict):
            for dir in dirs:
                out += _resc(os.path.join(fp, dir))
            out += [ os.path.join(fp, f) for f in files]
        return out
    else:
        return [dict]

def setEnv(home_path='/home/hadoop'):
    if 'SPARK_HOME' not in os.environ or os.environ['SPARK_HOME'] != '/usr/lib/spark':
        os.system("sudo sed -i -e '$a\export SPARK_HOME=/usr/lib/spark' {emr_home}/.bashrc".format(emr_home=home_path))
        os.environ['SPARK_HOME'] = '/usr/lib/spark'
    if 'HADOOP_HOME' not in os.environ or os.environ['HADOOP_HOME'] != '/usr':
        os.system("sudo sed -i -e '$a\export HADOOP_HOME=/usr' {emr_home}/.bashrc".format(emr_home=home_path))
        os.environ['HADOOP_HOME'] = '/usr'
    if 'PYSPARK_PYTHON' not in os.environ or os.environ['PYSPARK_PYTHON'] != '/usr/bin/python3':
        os.system(" sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh ")
        os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

def get_files_under_dates(data_file, filter_func=None):
    _data_file, _args_date = data_file.rsplit('/', 1)
    if ':' in _args_date:
        data_file = _data_file
        startdate, enddate = _args_date.rsplit(':', 1)
        # data_file, startdate = data_file.rsplit('/', 1)
        datelist = getDateInterval(startdate, enddate)
        fileList = reduce(lambda x, y: x + y, [
            _resc(os.path.join(data_file, date, file)) for date in datelist
            for file in tf.io.gfile.listdir(os.path.join(data_file, date))
            ])
    else:
        l = [_resc(os.path.join(data_file, file_dict)) for file_dict in tf.io.gfile.listdir(data_file)]
        fileList = reduce(lambda x, y: x+y, l)

    if filter_func is not None:
        fileList = list(filter(filter_func, fileList))
    return fileList

def getDateInterval(sd, ed):
    interval = []
    sd = dt.datetime.strptime(sd, "%Y-%m-%d")
    ed = dt.datetime.strptime(ed, "%Y-%m-%d")
    while sd <= ed:
        interval.append(sd.strftime("%Y-%m-%d"))
        sd += dt.timedelta(days=1)
    return interval

def elapse_time(start_time):
    return round((time.time() - start_time) / 60)


def print_config(conf_dict):
    train_conf = conf_dict['mission']['train']
    print("Using train config:")
    for k, v in train_conf.items():
        print('{}: {}'.format(k, v))

    model_conf = conf_dict['model']
    print("Using model config:")
    for k, v in model_conf.items():
        print('{}: {}'.format(k, v))

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.cast(labels, dtype=tf.float32)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.math.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.math.log(1-y_pred)
    L = tf.reduce_mean(input_tensor=L)
    return L

def tensorflow_save_parameters_with_partition(sess, save_path, out_type='txt'):
    if save_path.startswith('s3'):
        import boto3
        if tf.gfile.Exists('/tmp/dl_rank'):
            tf.gfile.DeleteRecursively('/tmp/dl_rank')
        tf.gfile.MakeDirs('/tmp/dl_rank')
        s3 = boto3.resource('s3')
    save_path = os.path.join(save_path, out_type)
    if tf.gfile.Exists(save_path):
        tf.gfile.DeleteRecursively(save_path)
    variables = sorted(sess.graph.get_collection(tf.GraphKeys.VARIABLES), key=lambda var: var.name)
    variables_names = [var.name for var in variables]
    var_dict = defaultdict(list)
    for idx, var_name in enumerate(variables_names):
        # var_name, tail = var_name.rsplit('/', 1)
        if '/' in var_name and re.match('part_\d+:\d+', var_name.rsplit('/', 1)[1]):
            save_name = var_name.rsplit('/', 1)[0]
            var_dict[save_name].append(variables[idx])
        else:
            save_name = var_name.split(':')[0]
            var_dict[save_name].append(variables[idx])
    for name, tensors in var_dict.items():
        file_name = os.path.join(save_path, name)
        file_path = file_name.rsplit('/', 1)[0]
        if not tf.gfile.Exists(file_path):
            tf.gfile.MakeDirs(file_path)
        merge_tensor = sess.run(tensors)
        if len(merge_tensor) > 1:
            merge_tensor = np.concatenate(merge_tensor, axis=0)
        else:
            merge_tensor = merge_tensor[0]
            if merge_tensor.ndim == 0:
                merge_tensor = np.expand_dims(merge_tensor, 0)
        if save_path.startswith('s3'):
            local_path = os.path.join('/tmp/dl_rank', name)
            _, _, bucketname, emr_path = file_name.split('/', 3)
            if not tf.gfile.Exists(local_path.rsplit('/', 1)[0]):
                tf.gfile.MakeDirs(local_path.rsplit('/', 1)[0])
            if out_type == 'npy':
                np.save(local_path+'.npy', merge_tensor)
                s3.Bucket(bucketname).upload_file(local_path+'.npy', emr_path+'.npy')
            else:
                np.savetxt(local_path+'.txt', merge_tensor)
                s3.Bucket(bucketname).upload_file(local_path+'.txt', emr_path+'.txt')
            pass
        else:
            if out_type == 'npy':
                np.save(file_name, merge_tensor)
            else:
                np.savetxt(file_name, merge_tensor)
        del merge_tensor

def reduceF(f, type):
    '''

    :param f: [bs, n, S], reduce axis 1
    :param type: combiner type, such as sum, mean, sqrtn, max, none
    :return: [bs, 1, S]
    '''
    if type == 'none':
        pass
    elif type == 'sum':
        f = tf.reduce_sum(f, axis=1, keep_dims=True)
    elif type == 'mean':
        f = tf.reduce_mean(f, axis=1, keep_dims=True)
    elif type == 'sqrtn':
        mean = tf.reduce_mean(f, axis=1, keep_dims=True)
        f = tf.sqrt(tf.reduce_sum(tf.square(f - mean), axis=1, keep_dims=True))
    elif type == 'max':
        f = tf.reduce_max(f, axis=1, keep_dims=True)
    else:
        assert False
    return f

def walks3(path):
    path = path.strip('/') + '/'
    out = subprocess.check_output(['aws', 's3', 'ls', path]).decode().split('\n')
    dirs = []
    files = []
    for row in out:
        if row == '' or row[-1] == ' ':
            continue
        row = row.strip()
        if row.startswith('PRE'):
            dirs.append(row.rsplit(' ', 1)[1])
        else:
            files.append(row.rsplit(' ', 1)[1])
    return path, dirs, files

def latest_checkpoint(src_s3):
    if gfile.Exists(src_s3):
        return tf.train.latest_checkpoint(src_s3)
    else:
        root, _, files = walks3(src_s3)
        maxStep = -1
        for file in files:
            num = file.split('model.ckpt-')[-1].split('.')[0]
            if num.isdigit():
                maxStep = max(maxStep, int(num))
        if maxStep<0:
            assert False, 'Cant find checkpoint under {}'.format(src_s3)
        else:
            latest_checkpoint = 'model.ckpt-'+str(maxStep)
            return os.path.join(src_s3, latest_checkpoint)



def copyS3toHDFS(sc, src_s3, dest_hdfs):
    lastest = latest_checkpoint(src_s3)
    root, ckpt_name = lastest.rsplit('/', 1)
    model_ckpt = [f for f in walks3(src_s3)[2] if ckpt_name in f]
    model_ckpt += ['checkpoint', 'graph.pbtxt']
    abs_ckpt = [root+'/'+f for f in model_ckpt]
    dir_process = subprocess.Popen(['hadoop', 'fs', '-mkdir', '{dest_file}'.format(dest_file=dest_hdfs)])
    dir_process.wait()
    process = [subprocess.Popen(['hadoop', 'fs', '-cp', '{src_file}'.format(src_file=ckpt_file), '{dest_file}'
                                .format(dest_file=dest_hdfs)]) for ckpt_file in abs_ckpt]
    _ = [p.wait() for p in process]
    print('finish copy model from {} to {}'.format(src_s3, dest_hdfs))
    hadoop = sc._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    path = hadoop.fs.Path('/user/hadoop/dl_rank')
    subfiles = [f for f in fs.get(conf).listStatus(path)]
    first_ckpt_hdfs = subfiles[0].getPath().toString()
    ckpt_dir = first_ckpt_hdfs.rsplit('/', 1)[0]
    return ckpt_dir