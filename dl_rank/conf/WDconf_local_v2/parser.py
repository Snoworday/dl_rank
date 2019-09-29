import tensorflow as tf
import os
import shutil
import numpy as np
import time
from collections import OrderedDict
from dl_rank import BaseParser

try:
    from ..BaseParser import BaseParser
except:
    from dl_rank import BaseParser
import yaml
import logging
from functools import reduce
class Parser(BaseParser):
    pass


    # @staticmethod
    # def secondary_parse_fn(sparse_emb, deep_emb, dense_emb, mask, model_struct):
    #     wide = ['cid', 'click_cross', 'order_cross', 'gmv', 'ctr']
    #     itemid = 'pid'
    #     sequence = 'click'
    #     tfprint = tf.print('cid', deep_emb['cid'], 'click_cross', deep_emb['click_cross'], 'order', deep_emb['order_cross'])
    #     with tf.control_dependencies([]):
    #         sparse_features = tf.concat([tf.squeeze(deep_emb[w], axis=1) for w in wide], axis=1)
    #     itemid_emb = deep_emb[itemid]
    #     sequence_emb = deep_emb[sequence]
    #     sequence_token_mask = mask['click']
    #     itemid_token_mask = tf.ones([tf.shape(mask['click'])[0], 1])
    #
    #     return sparse_features, itemid_emb, sequence_emb, sequence_token_mask, itemid_token_mask