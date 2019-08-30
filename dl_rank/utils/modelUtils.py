#config=utf-8
import tensorflow as tf
import os
import shutil
import numpy as np
import time
import argparse
import sys
import logging
# get TF logger

import datetime as dt

def open_log(path):
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(path,'dl_rank_tf.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    # os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def tensorflow_get_weights(sess):
    vs = {v.name: v for v in tf.compat.v1.get_default_graph().get_collection(
        tf.compat.v1.GraphKeys.VARIABLES)}  # tf.compat.v1.trainable_variables()}
    name_values = sess.run(vs)
    return name_values


def tensorflow_set_weights(sess, weights):
    assign_ops = []
    feed_dict = {}
    vs = {v.name: v for v in tf.compat.v1.get_default_graph().get_collection(tf.compat.v1.GraphKeys.VARIABLES)}
    for vname, v in vs.items():
        value = np.asarray(weights[vname])
        assign_placeholder = tf.compat.v1.placeholder(v.dtype, shape=v.shape)
        assign_op = v.assign(assign_placeholder)
        assign_ops.append(assign_op)
        feed_dict[assign_placeholder] = value
    sess.run(assign_ops, feed_dict=feed_dict)

def convertDate(date, delta):
    if date == '':
        return date
    date = date.split(':')[-1]
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = (date + dt.timedelta(days=delta)).strftime("%Y-%m-%d")
    return date

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


def print_config(load_conf):
    train_conf = load_conf('train.yaml')['train']
    print("Using train config:")
    for k, v in train_conf.items():
        print('{}: {}'.format(k, v))

    model_conf = load_conf('model.yaml')
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



def _input_parser(s):
  l = s.split('@')
  if len(l)==4:
      logging.info('gooddata')
  else:
      logging.info('baddata')
  assert len(l) == 4
  out = '@'.join(l)
  return out