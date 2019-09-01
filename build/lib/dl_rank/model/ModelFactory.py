import tensorflow as tf
from . import model_dict
import logging

class modelFactory(object):
    @staticmethod
    def build(train_conf, model_conf, mode):
        model_type = train_conf['model_type']
        Model = model_dict[model_type](model_conf, mode)
        return Model


