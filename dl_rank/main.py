import os
import sys
if sys.path[0]!= '':
    os.chdir(sys.path[0])
try:
    from .model.ModelFactory import modelFactory
    from .utils.modelUtils import getDateInterval,convertDate,print_config,elapse_time,tensorflow_set_weights,tensorflow_get_weights,open_log
    from .conf import conf_parser
except:
    from model.ModelFactory import modelFactory
    from utils.modelUtils import getDateInterval,convertDate,print_config,elapse_time,tensorflow_set_weights,tensorflow_get_weights,open_log
    from conf import conf_parser
import tensorflow as tf
from tensorflow.python.framework import graph_util
import json
import shutil
import numpy as np
import time
import argparse
import logging
from functools import reduce
tf.compat.v1.disable_eager_execution()

DEBUG = False
# with open('thereistag.txt', 'w') as f:
#     f.write('hh')
# get TF logger




class EstimatorManager(object):
    def __init__(self, parser, args, spark=None):
        self.parser = parser
        self.model_out_format = self.parser.model_out_format
        self.all_conf = parser.load_all_conf('model', 'feature', 'train', 'vocabulary')
        self.model_conf = self.all_conf['model']
        self.feature_conf = self.all_conf['feature']
        self.train_conf = self.all_conf['train']['train']
        self.run_conf = self.all_conf['train']['runconfig']
        self.vocabulary_conf = self.all_conf['vocabulary']

        self.model = modelFactory.build(self.train_conf, self.model_conf, args['mode'])
        self.model.set_embedding_parser(self.parser.model_input_parse_fn)

        self.spark = spark
        self.args = args
        pass

    def input_fn(self):
        _args_mode = self.args['mode']
        _args_date = self.args['date']
        def wrapper(data_file, parse_fn, mode, batch_size, TOTAL_WORKERS, WORKER_INDEX, epochs, num_parallel_calls,
                     shuffle_buffer_size):
            tf.compat.v1.logging.info("Parsing input file: {}".format(data_file))
            if _args_date == '':
                datelist = tf.io.gfile.listdir(data_file)
                fileList = reduce(lambda x, y: x + y, [
                    [os.path.join(data_file, date, file) for file in tf.io.gfile.listdir(os.path.join(data_file, date))]
                    for date in datelist if not date.startswith('.')])
            elif ':' in _args_date and mode == 'train':
                data_file, enddate = data_file.rsplit(':', 1)
                data_file, startdate = data_file.rsplit('/', 1)
                datelist = getDateInterval(startdate, enddate)
                fileList = reduce(lambda x, y: x + y, [
                    [os.path.join(data_file, date, file) for file in tf.io.gfile.listdir(os.path.join(data_file, date))]
                    for date in datelist])
            else:
                fileList = [os.path.join(data_file, f) for f in tf.io.gfile.listdir(data_file)]
            fileList = list(filter(lambda f_name: f_name.split('.')[-1] == 'txt', fileList))
            dataset = tf.data.TextLineDataset(fileList)
            if TOTAL_WORKERS > 1:
                dataset = dataset.shard(TOTAL_WORKERS, WORKER_INDEX)
            if _args_mode == 'train':
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=1234567)
                dataset = dataset.repeat(epochs)
            dataset = dataset.map(parse_fn(isPred=_args_mode == 'infer'),
                                  num_parallel_calls=num_parallel_calls)
            dataset = dataset.prefetch(2 * batch_size)
            dataset = dataset.batch(batch_size)
            return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        return wrapper

    def build_estimator(self):
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=0,
                                intra_op_parallelism_threads=0,
                                log_device_placement=True,
                                allow_soft_placement=True
                                # device_count = {"GPU": 1}  # limit to GPU usage
                                )
        if tf.__version__[0]=='2':
            if self.args['useps']:
                self.run_conf.update({'train_distribute': tf.distribute.experimental.ParameterServerStrategy()})
            else:
                self.run_conf.update({'train_distribute': tf.distribute.experimental.MultiWorkerMirroredStrategy()})
        else:
            if not self.args['useps']:
                self.run_conf.update({'train_distribute': tf.contrib.distribute.CollectiveAllReduceStrategy()})
        run_config = tf.estimator.RunConfig(**self.run_conf).replace(session_config=config)

        params = {
            'feature_conf': self.feature_conf,
            'model_conf': self.model_conf,
            'vocabulary_conf': self.vocabulary_conf
        }
        model_dir = self.train_conf['model_dir']

        return tf.estimator.Estimator(
            model_dir=model_dir,
            model_fn=self.model_fn,
            params=params,
            config=run_config
        )

    def export_model(self, save_path='', remove_subdir=False):
        save_path = self.train_conf['graph_dir'] if not save_path else save_path
        org_dir_set = set(tf.gfile.ListDirectory(save_path))
        self.estimator.export_saved_model(save_path, self.parser.serving_parse_fn(), as_text=False)
        sub_dir_name = (set(tf.gfile.ListDirectory(save_path)) - org_dir_set).pop()
        sub_dir_path = os.path.join(save_path, sub_dir_name)
        for fpath, dirs, files in tf.gfile.Walk(sub_dir_path, in_order=True):
            for dir in dirs:
                old_path = os.path.join(fpath, dir)
                new_path = os.path.join(*old_path.split(sub_dir_name+'/'))
                if not tf.gfile.IsDirectory(new_path):
                    tf.gfile.MkDir(new_path)
            for file in files:
                old_path = os.path.join(fpath, file)
                new_path = os.path.join(*old_path.split(sub_dir_name+'/'))
                if tf.gfile.Exists(new_path):
                    tf.gfile.Remove(new_path)
                tf.gfile.Copy(old_path, new_path)
        if remove_subdir:
            tf.gfile.DeleteRecursively(sub_dir_path)


    def export_model_online(self, save_path='', output_node_names=None, input_node_map=None, from_pb=False):
        save_path = self.train_conf['online_graph_dir'] if save_path == '' else save_path
        input_node_map = self.model.placeholder_map if not input_node_map else input_node_map
        graph_pb_path = self.train_conf['graph_dir'] if from_pb else ''
        checkpoint_path = self.train_conf['model_dir'] if not from_pb else ''
        if not output_node_names:
            output_node_names = self.model.output_node_name
        self._export_model_online(save_predict_dir=save_path,
                                  output_node_names=output_node_names,
                                  replace_map=input_node_map,
                                  graph_pb_path=graph_pb_path,
                                  checkpoint_path=checkpoint_path)

    @staticmethod
    def _export_model_online(save_predict_dir, output_node_names, replace_map, graph_pb_path='', checkpoint_path='',
                      output_variable_name='data.ckpt', output_graph_name='model.pb'):
        assert graph_pb_path != '' or checkpoint_path != '', 'Give me ur graph and vars, OAO'
        org_graph = tf.Graph()
        new_graph = tf.Graph()
        with tf.Session(graph=org_graph) as sess:
            if graph_pb_path != '':
                org_meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [
                    tf.compat.v1.saved_model.tag_constants.SERVING], graph_pb_path)
                weights = tensorflow_get_weights(sess)
            else:
                org_meta_graph_def = tf.train.latest_checkpoint(checkpoint_path) + '.meta'
                _ = tf.train.import_meta_graph(org_meta_graph_def)

        with tf.Session(graph=new_graph) as sess:
            input_map = dict()
            for org_name in replace_map:
                org_tensor = org_graph.get_tensor_by_name(org_name + ':0')
                input_map[org_name + ':0'] = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape,
                                                            name=replace_map[org_name])
            _ = tf.train.import_meta_graph(org_meta_graph_def, input_map=input_map, return_elements=output_node_names)
            if checkpoint_path != '':
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_path)
            else:
                tensorflow_set_weights(sess, weights)
                saver = tf.train.Saver()
            saver.save(sess, save_path=os.path.join(save_predict_dir, output_variable_name))
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names
            )
            output_graph_path = os.path.join(save_predict_dir, output_graph_name)
            with tf.gfile.GFile(output_graph_path, 'wb') as f:
                f.write(output_graph_def.SerializeToString())

    def model_fn(self, features, labels, mode, params):
        """Model function used in the estimator.
        Args:
            features (Tensor): Input features to the model.
            labels (Tensor): Labels tensor for training and evaluation.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.
        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if mode == tf.estimator.ModeKeys.PREDICT:
            ids = {id:features.pop(id) for id in self.model_out_format if id!='out'}
            # ids = {key: features[key] for key in features if key != 'features'}
            predictions = self.model.forward(features, params=params, is_training=is_training)
            predictions_out = self.model.get_predictions_out(features, predictions, ids, self.model_out_format)
            # export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions_out)}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions_out) #, export_outputs=export_outputs)

        predictions = self.model.forward(features, params=params, is_training=is_training)
        if len(predictions.get_shape().as_list()) == 1:
            predictions = predictions[:, np.newaxis]
        if len(labels.get_shape().as_list()) == 1:
            labels = labels[:, np.newaxis]
        loss = self.model.get_loss(labels, predictions)
        eval_metric_ops = self.model.get_eval_metric_ops(labels, predictions)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
        if DEBUG:
            compare = tf.concat([tf.as_string(labels), tf.as_string(predictions), tf.as_string(tf.cast(labels, dtype=tf.float32)-predictions)], axis=1)
            hook = \
                [tf.estimator.LoggingTensorHook({'result': compare},
                                           every_n_iter=1)]
        else:
            hook = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            self.model.add_summary(labels, predictions, eval_metric_ops)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = self.model.get_train_op_fn(loss, params)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=hook)

    @property
    def estimator(self):
        if hasattr(self, '_estimator'):
            return self._estimator
        else:
            self._estimator = self.build_estimator()
            return self._estimator

    def train_and_eval(self, parse_fn):
        train_epochs = self.train_conf['train_epochs']
        max_steps = self.train_conf['max_steps']
        steps = self.train_conf['steps']
        train_data = os.path.join(self.train_conf['train_data'], self.args['date'])
        eval_data = os.path.join(self.train_conf['eval_data'], convertDate(self.args['date'], 1))
        batch_size = self.train_conf['batch_size']
        input_fn = self.input_fn()
        # predict_batch(train_data, parse_fn)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(train_data, parse_fn, tf.estimator.ModeKeys.TRAIN, batch_size, self.args['tw'], self.args['wi'],
                                                       self.train_conf['train_epochs'], self.train_conf['num_parallel_calls'], self.train_conf['shuffle_buffer_size']), max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(eval_data, parse_fn, tf.estimator.ModeKeys.EVAL, batch_size, self.args['tw'], self.args['wi'],
                                                       self.train_conf['eval_epochs'], self.train_conf['num_parallel_calls'], self.train_conf['shuffle_buffer_size']), steps=steps, throttle_secs=600)

        t0 = time.time()

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        self.export_model()
        tf.compat.v1.logging.info(
            '<EPOCH {}>: Finish evaluation {}, take {} mins'.format(train_epochs + 1, eval_data, elapse_time(t0)))

    def test(self, parse_fn):
        train_epochs = self.train_conf['train_epochs']
        max_steps = self.train_conf['max_steps']
        steps = self.train_conf['steps']
        pred_data = os.path.join(self.train_conf['pred_data'], self.args['date'])
        batch_size = self.train_conf['batch_size']
        epochs_per_eval = self.train_conf['epochs_per_eval']
        input_fn = self.input_fn()

        out = self.estimator.predict(lambda: input_fn(pred_data, parse_fn, tf.estimator.ModeKeys.PREDICT, 8,
                                                                       self.args['tw'], self.args['wi'], 1, self.train_conf['num_parallel_calls'], self.train_conf['shuffle_buffer_size']))
        for _ in range(100000):
            a = next(out)
            print(a)

    def predict(self, parse_fn):
        import pandas as pd
        from pyspark.sql.functions import pandas_udf, PandasUDFType, col
        from pyspark.sql.types import ArrayType, FloatType
        from tensorflow.python.framework import tensor_util

        graph_dir = self.train_conf['graph_dir']
        save_dir = os.path.join(self.train_conf['result_dir'], self.args['date'])
        model_out_format = self.parser.model_out_format
        batch_size = self.train_conf['batch_size']
        sc = self.spark.sparkContext
        sess = tf.compat.v1.Session()
        model = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], graph_dir)
        weights = tensorflow_get_weights(sess)

        bc_weights = sc.broadcast(weights)
        bc_model = sc.broadcast(model)
        sess.close()
        pred_data = os.path.join(self.train_conf['pred_data'], self.args['date'])
        df = self.spark.read.text(pred_data).rdd.toDF()
        # df = spark.read.text(pred_data).rdd.filter(lambda row: len(row[0].split('@'))==3).toDF()
        def _prediction_batch(record_batch):
            sess = tf.compat.v1.Session()
            input_batch = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            dataset = tf.data.Dataset.from_tensor_slices(input_batch)  ###
            dataset = dataset.map(parse_fn(isPred=True, tail=':0'), num_parallel_calls=16).batch(batch_size)
            iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            data_batch = iterator.get_next()

            tf.compat.v1.train.import_meta_graph(bc_model.value)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            tensorflow_set_weights(sess, bc_weights.value)
            sess.run(iterator.initializer, feed_dict={input_batch: record_batch})
            out_tensor = sess.graph.get_tensor_by_name('out:0')
            result = []
            try:
                while True:
                    batch = sess.run(data_batch)
                    preds = sess.run(out_tensor, batch)
                    id_preds = np.concatenate([batch['{}:0'.format(elem)].astype(np.int32)[:, np.newaxis]
                                               if elem != 'out' else preds for elem in model_out_format], axis=1)
                    # id_preds = np.concatenate([batch['pid1:0'].astype(np.int32)[:, np.newaxis], batch['pid2:0'].astype(np.int32)[:, np.newaxis], preds], axis=1)
                    result = result + list(id_preds)
            except tf.errors.OutOfRangeError:
                pass
            return pd.Series(result)

        # input_batch = df.toPandas().loc[:, 'value']
        # dataset = _prediction_batch(input_batch)
        predict_batch_udf = pandas_udf(returnType=ArrayType(FloatType()), functionType=PandasUDFType.SCALAR)(_prediction_batch)
        predictions = df.select(predict_batch_udf(col("value")).alias('prediction'))
        predictions.rdd.map(lambda arrfloat: tuple([int(arrfloat[0][i]) if i<len(model_out_format)-1 else arrfloat[0][i] for i in range(len(arrfloat[0]))])).saveAsTextFile(save_dir)
        print('finish')


def run(mode, conf, useSpark, date='', tw=1, wi=1, useps=True, logpath='', **kwargs):
    open_log(logpath)
    # _args = args
    if useSpark or mode in ['infer', 'test']:
        from pyspark.sql import SparkSession
        spark = SparkSession \
            .builder \
            .appName('dl_rank') \
            .getOrCreate()
    else:
        spark = None
    dataparser = conf_parser(conf, useSpark=useSpark)
    print_config(dataparser.load_conf)
    # if not train_conf['keep_train']:
    #     shutil.rmtree(model_dir, ignore_errors=True)
    #     print("remove model directory: {}".format(model_dir))
    _args = {'mode': mode, 'date': date, 'tw': tw, 'wi': wi, 'useps': useps}
    eM = EstimatorManager(dataparser, _args, spark)
    tf.compat.v1.logging.info("Build estimator: {}".format(eM.estimator))
    if mode == 'train':
        eM.train_and_eval(dataparser.parse_fn)
    elif mode == 'export':
        eM.export_model(**kwargs)
    elif mode == 'export_online':
        eM.export_model_online(**kwargs)
    elif mode == 'test':
        eM.args['mode'] = 'infer'
        eM.test(dataparser.parse_fn)
    else:
        eM.predict(dataparser.parse_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tw', type=int, help='number of machines', default=1)
    parser.add_argument('--wi', type=int, help='worker idx', default=1)
    parser.add_argument('--conf', type=str, help='switch I2Iconf_test')
    parser.add_argument('--date', type=str, help='date of data', default='')
    parser.add_argument('--mode', type=str, help='[train|infer|test|export|export_online]', default='train')
    parser.add_argument('--useps', type=bool,
                        help='if True: use parameterServerStrategy else: use MultiWorkerMirroredStrategy', default=True)
    parser.add_argument('--tfconfig', type=str, help='json string of TF_CONFIG', default='')
    parser.add_argument('--useSpark', type=bool, help='use spark-submit when infer', default=False)
    parser.add_argument('--logpath', type=str, help='path to write TF_CONFIG/train.log', default='')
    args = parser.parse_args()
    if args.tfconfig != '':
        os.environ['TF_CONFIG'] = args.tfconfig.replace('_', '"').replace('*', ',')
        with open(os.path.join(args.logpath, 'tf_config.txt'), 'w') as f:
            f.write(os.environ['TF_CONFIG'])
    run(args.mode, args.conf, args.useSpark, args.date, args.tw, args.wi, args.useps, args.logpath)
