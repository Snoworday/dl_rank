from __future__ import absolute_import

import os
import datetime as dt
import types
try:
    from dl_rank.conf import conf_parser
    from dl_rank.model.ModelFactory import modelFactory
    from dl_rank.utils.modelUtils import convertDate,print_config,elapse_time,open_log,setEnv,tensorflow_save_parameters_with_partition,copyS3toHDFS
except:
    from conf import conf_parser
    from model.ModelFactory import modelFactory
    from utils.modelUtils import convertDate,print_config,elapse_time,open_log,setEnv,tensorflow_save_parameters_with_partition,copyS3toHDFS
import tensorflow as tf
from tensorflow.python.framework import graph_util
import shutil
import subprocess
import numpy as np
import time
import argparse
import logging
from functools import reduce
tf.compat.v1.disable_eager_execution()

DEBUG = False


class EstimatorManager(object):
    def __init__(self, parser, args, spark=None):
        self.parser = parser

        self.all_conf = parser.load_all_conf('model', 'feature', 'mission', 'vocabulary', 'schema')
        self.model_conf = self.all_conf['model']
        self.feature_conf = self.all_conf['feature']
        self.train_conf = self.all_conf['mission']['train']
        self.run_conf = self.all_conf['mission']['runconfig']
        self.vocabulary_conf = self.all_conf['vocabulary']
        self.feature_list = self.all_conf['schema']
        self.spark = spark
        self.args = args
        pass

    def initModel(self, exPath=None):
        modelFactory.external_path = exPath
        self.model = modelFactory.build(self.train_conf, self.model_conf, self.args['mode'])
        self.model.set_embedding_parser(self.parser.model_input_parse_fn)

    def input_fn(self):
        _args_mode = self.args['mode']
        filter_func = self.parser.data_file_filter

        def wrapper(parse_fn, mode, batch_size, data_source, TOTAL_WORKERS=1, WORKER_INDEX=1, epochs=1, num_parallel_calls=1,
                    shuffle_buffer_size=1024, tail=''):
            if isinstance(data_source, str):
                tf.compat.v1.logging.info("Parsing input file: {}".format(data_source))
                fileList = Utils.get_files_under_dates(data_source, filter_func)
                dataset = tf.data.TextLineDataset(fileList)
                if TOTAL_WORKERS > 1:
                    dataset = dataset.shard(TOTAL_WORKERS, WORKER_INDEX)
                if _args_mode == 'train':
                    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=1234567)
                    dataset = dataset.repeat(epochs)
                dataset = dataset.batch(batch_size)
            elif isinstance(data_source, types.FunctionType):
                dataset = tf.data.Dataset.from_generator(data_source, (tf.string), (tf.TensorShape([None])))
            else:   # placeholder
                dataset = tf.data.Dataset.from_tensor_slices(data_source)
                dataset = dataset.batch(batch_size)
            dataset = dataset.map(parse_fn(isPred=_args_mode != 'train', tail=tail),
                                  num_parallel_calls=num_parallel_calls)
            dataset = dataset.prefetch(2 * batch_size)
            if _args_mode != 'infer' or isinstance(data_source, types.FunctionType):
                return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
            else:
                return tf.data.make_initializable_iterator(dataset).get_next()
        return wrapper

    def build_estimator(self):
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=0,
                                intra_op_parallelism_threads=0,
                                log_device_placement=True,
                                allow_soft_placement=True
                                # device_count = {"GPU": 1}  # limit to GPU usage
                                )
        if tf.__version__[0] == '2':
            if self.args['ps'] > 0:
                self.run_conf.update({'train_distribute': tf.distribute.experimental.ParameterServerStrategy()})
            else:
                self.run_conf.update({'train_distribute': tf.distribute.experimental.MultiWorkerMirroredStrategy()})
        else:
            if self.args['ps'] > 0:
                pass
                # self.run_conf.update({'train_distribute': tf.contrib.distribute.ParameterServerStrategy()})
            else:
                self.run_conf.update({'train_distribute': tf.contrib.distribute.CollectiveAllReduceStrategy()})

        run_config = tf.estimator.RunConfig(**self.run_conf).replace(session_config=config)

        params = {
            'feature_conf': self.feature_conf,
            'feature_list': self.feature_list,
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

    def export_model(self, save_path=None, remove_subdir=False, pbtxt=True):
        save_path = self.train_conf['graph_dir'] if save_path is None else save_path
        if not tf.gfile.IsDirectory(save_path):
            tf.gfile.MakeDirs(save_path)
        org_dir_set = set(tf.gfile.ListDirectory(save_path))
        self.estimator.export_saved_model(save_path, self.parser.serving_parse_fn(self.model.out_node_names), as_text=pbtxt)
        sub_dir_name = (set(tf.gfile.ListDirectory(save_path)) - org_dir_set).pop()
        sub_dir_path = os.path.join(save_path, sub_dir_name)
        for fpath, dirs, files in tf.gfile.Walk(sub_dir_path, in_order=True):
            for dir in dirs:
                old_path = os.path.join(fpath, dir)
                new_path = os.path.join(*old_path.split(sub_dir_name+'/'))
                if not tf.gfile.IsDirectory(new_path):
                    tf.gfile.MakeDirs(new_path)
            for file in files:
                old_path = os.path.join(fpath, file)
                new_path = os.path.join(*old_path.split(sub_dir_name+'/'))
                if tf.gfile.Exists(new_path):
                    tf.gfile.Remove(new_path)
                tf.gfile.Copy(old_path, new_path)
        if remove_subdir:
            tf.gfile.DeleteRecursively(sub_dir_path)

    def export_model_online(self, save_path=None, out_node_name=None, input_node_map=None, output_data_type='txt', from_pb=True):
        save_path = self.train_conf['online_graph_dir'] if save_path is None else save_path
        input_node_map = self.model.placeholder_map if not input_node_map else input_node_map
        graph_pb_path = self.train_conf['graph_dir'] if from_pb else ''
        checkpoint_path = self.train_conf['model_dir'] if not from_pb else ''
        if not out_node_name:
            out_node_name = self.model.out_node_names
        self._export_model_online(save_predict_dir=save_path,
                                  out_node_name=out_node_name,
                                  replace_map=input_node_map,
                                  output_data_type=output_data_type,
                                  graph_pb_path=graph_pb_path,
                                  checkpoint_path=checkpoint_path)

    @staticmethod
    def _tf_weights_ops(set_or_get):
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

        def tensorflow_get_weights(sess):
            vs = {v.name: v for v in tf.compat.v1.get_default_graph().get_collection(
                tf.compat.v1.GraphKeys.VARIABLES)}  # tf.compat.v1.trainable_variables()}
            name_values = sess.run(vs)
            return name_values

        if set_or_get == 'set':
            return tensorflow_set_weights
        if set_or_get == 'get':
            return tensorflow_get_weights

    @staticmethod
    def _export_model_online(save_predict_dir, out_node_name, replace_map, graph_pb_path='', checkpoint_path='',
                             output_data_type='txt', output_graph_name='model.pb'):
        assert graph_pb_path != '' or checkpoint_path != '', 'Give me ur graph and vars, OAO'
        if not tf.gfile.IsDirectory(save_predict_dir):
            tf.gfile.MakeDirs(save_predict_dir)
        org_graph = tf.Graph()
        new_graph = tf.Graph()
        # read graph
        with tf.Session(graph=org_graph) as sess:
            if graph_pb_path != '':
                org_meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [
                    tf.compat.v1.saved_model.tag_constants.SERVING], graph_pb_path)
                weights = EstimatorManager._tf_weights_ops('get')(sess)
            else:
                org_meta_graph_def = tf.train.latest_checkpoint(checkpoint_path) + '.meta'
                old_saver = tf.train.import_meta_graph(org_meta_graph_def, clear_devices=True)
                old_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            # EstimatorManager._tf_weights_ops('save')()
            if output_data_type in ['npy', 'txt']:
                tensorflow_save_parameters_with_partition(sess, save_predict_dir, output_data_type)
            elif output_data_type == 'ckpt':
                old_saver.save(sess, save_path=os.path.join(save_predict_dir, 'ckpt', 'data.ckpt'))

        # set placeholder & write graph->pb/variables->ckpt
        with tf.Session(graph=new_graph) as sess:
            input_map = dict()
            for org_name in replace_map:
                org_tensor = org_graph.get_tensor_by_name(org_name + ':0')
                input_map[org_name + ':0'] = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape,
                                                            name=replace_map[org_name])
            _ = tf.train.import_meta_graph(org_meta_graph_def,
                                           input_map=input_map, return_elements=out_node_name, clear_devices=True)
            if checkpoint_path != '':
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            else:
                EstimatorManager._tf_weights_ops('set')(sess, weights)
                saver = tf.train.Saver()
            # if output_data_type == 'ckpt':
            #     saver.save(sess, save_path=os.path.join(save_predict_dir, 'ckpt', 'data.ckpt'))
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                out_node_name
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
            ids = {id:features.pop(id) for id in self.parser.model_out_format if id[:-1] not in self.model.out_node_names}
            # ids = {key: features[key] for key in features if key != 'features'}
            predictions = self.model.forward(features, params=params, is_training=is_training)
            predictions_out = self.model.get_predictions_out(features, predictions, ids, self.parser.model_out_format)
            # export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions_out)}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions_out) #, export_outputs=export_outputs)

        predictions = self.model.forward(features, params=params, is_training=is_training)
        if predictions is not None and predictions.shape.ndims == 1:
            predictions = predictions[:, np.newaxis]
        if labels is not None and labels.shape.ndims == 1:
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
        eval_data = os.path.join(self.train_conf['eval_data'], self.args['date'])# convertDate(self.args['date'], 1))
        batch_size = self.train_conf['batch_size']
        input_fn = self.input_fn()
        # predict_batch(train_data, parse_fn)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(parse_fn, tf.estimator.ModeKeys.TRAIN, batch_size, train_data, self.args['tw'], self.args['wi'],
                                                       self.train_conf['train_epochs'], self.train_conf['num_parallel_calls'], self.train_conf['shuffle_buffer_size']), max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(parse_fn, tf.estimator.ModeKeys.EVAL, batch_size, eval_data, self.args['tw'], self.args['wi'],
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

        out = self.estimator.predict(lambda: input_fn(parse_fn, tf.estimator.ModeKeys.PREDICT, 8, pred_data,
                                                                       self.args['tw'], self.args['wi'], 1, self.train_conf['num_parallel_calls'], self.train_conf['shuffle_buffer_size']))
        return out
        # for _ in range(100000):
        #     a = next(out)
        #     print(a)

    def predict(self, parse_fn, bake_path=''):
        import pandas as pd
        from pyspark.sql.functions import pandas_udf, PandasUDFType, col
        from pyspark.sql.types import ArrayType, FloatType, StringType

        graph_dir = self.train_conf['graph_dir']
        save_dir = os.path.join(self.train_conf['result_dir'], self.args['date'])
        model_out_format = self.parser.model_out_format
        out_node_names = self.model.out_node_names
        delim = self.parser.conf_dict['separator']['pred_out_delim']
        batch_size = self.train_conf['batch_size']
        sc = self.spark.sparkContext
        input_fn = self.input_fn()
        sess = tf.compat.v1.Session()
        if not tf.gfile.Exists(graph_dir):
            os.system('aws s3 cp {graph_dir} {save_path} --recursive'.format(graph_dir=graph_dir, save_path=os.path.join(bake_path, 'graph')))
            graph_dir = os.path.join(bake_path, 'graph')
        model = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], graph_dir)
        weights = EstimatorManager._tf_weights_ops('get')(sess)
        tensorflow_set_weights = EstimatorManager._tf_weights_ops('set')

        bc_weights = sc.broadcast(weights)
        bc_model = sc.broadcast(model)
        sess.close()
        pred_data = os.path.join(self.train_conf['pred_data'], self.args['date'])
        df = self.spark.read.text(Utils.get_files_under_dates(pred_data, self.parser.data_file_filter))
        def _prediction_batch(record_batch):
            tf.reset_default_graph()
            record_batch = record_batch.astype(np.str)
            sess = tf.compat.v1.Session()
            input_batch = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            data_batch = input_fn(parse_fn, 'infer', batch_size, input_batch, tail=':0')
            data_iter_init_op = sess.graph.get_operation_by_name('MakeIterator')    # get op of tf.data.make_initializable_iterator, by default name
            tf.compat.v1.train.import_meta_graph(bc_model.value)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            tensorflow_set_weights(sess, bc_weights.value)
            sess.run(data_iter_init_op, feed_dict={input_batch: record_batch})
            out_tensors = [sess.graph.get_tensor_by_name('{}:0'.format(name)) for name in out_node_names]
            result = []
            try:
                while True:
                    batch = sess.run(data_batch)
                    ids = dict()
                    for elem in model_out_format:
                        if elem[:-1] not in out_node_names:
                            ids[elem] = np.expand_dims(batch.pop('{}:0'.format(elem)), 1)#[:, np.newaxis]
                    preds = sess.run(out_tensors, batch)
                    preds_dict = dict(zip(out_node_names, preds))
                    id_preds = np.concatenate([ids[elem]
                                               if elem[:-1] not in out_node_names else preds_dict[elem[:-1]].astype(np.str) for elem in model_out_format], axis=1)
                    result = result + list(id_preds)
            except tf.errors.OutOfRangeError:
                pass
            return pd.Series(result)

        # input_batch = df.toPandas().loc[:, 'value']
        # dataset = _prediction_batch(input_batch)
        predict_batch_udf = pandas_udf(returnType=ArrayType(StringType()), functionType=PandasUDFType.SCALAR)(_prediction_batch)
        predictions = df.select(predict_batch_udf(col("value")).alias('prediction'))
        if tf.gfile.Exists(save_dir):
            tf.gfile.DeleteRecursively(save_dir)
        predictions.rdd.map(lambda x: delim.join(x[0])).saveAsTextFile(save_dir)
        print('finish')

    def predict_TFoS(self, parse_fn, num_executor, num_ps, bake_path, rdma=True):
        import os
        import argparse
        import subprocess
        try:
            from dl_rank.tensorflowonspark import TFCluster
        except:
            from tensorflowonspark import TFCluster
        #----------add hadoop class to CLASSPATH for tf read hdfs----
        os.environ['CLASSPATH'] = subprocess.check_output('hadoop classpath --glob', shell=True).decode('utf-8')

        sc = self.spark.sparkContext
        input_fn = self.input_fn()
        batch_size = self.train_conf['batch_size']
        params = {
            'feature_conf': self.feature_conf,
            'feature_list': self.feature_list,
            'model_conf': self.model_conf,
            'vocabulary_conf': self.vocabulary_conf
        }
        is_training = False
        model = self.model
        parser = self.parser
        out_node_names = self.model.out_node_names
        delim = self.parser.conf_dict['separator']['pred_out_delim']
        ckpt_dir = self.train_conf['model_dir']
        if ckpt_dir.startswith('s3'):
            ckpt_dir = copyS3toHDFS(sc, ckpt_dir, bake_path)
        tf_args = argparse.ArgumentParser()
        tf_args.rdma = rdma

        def map_fun(args, ctx):
            worker_num = ctx.worker_num
            job_name = ctx.job_name
            task_index = ctx.task_index
            cluster, server = ctx.start_cluster_server(0, args.rdma)
            tf_feed = ctx.get_data_feed(train_mode=False)
            decode_str_fn = np.vectorize(lambda x: x.decode())
            def rdd_generator():
                while not tf_feed.should_stop():
                    batch_data = tf_feed.next_batch(batch_size)
                    if len(batch_data) == 0:
                        return
                    yield batch_data
            if job_name == 'ps':
                server.join()
            elif job_name == 'worker':
                with tf.device(tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % task_index, cluster=cluster
                )):
                    features = input_fn(parse_fn, 'infer',  batch_size, rdd_generator)   # dict{features:tensor, }

                    ids = {id: features.pop(id) for id in parser.model_out_format if
                           id[:-1] not in out_node_names}
                    ids = {name: tf.expand_dims(tensor, 1) for name, tensor in ids.items()}

                    predictions = model.forward(features, params=params, is_training=is_training)
                    predictions_out = model.get_predictions_out(features, predictions, ids,
                                                                     parser.model_out_format)
                    global_step = tf.compat.v1.train.get_or_create_global_step()
                    saver = tf.compat.v1.train.Saver()
                    summary_op = tf.compat.v1.summary.merge_all()
                    init_op = tf.compat.v1.global_variables_initializer()

                    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=task_index==0,
                                                           scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),
                                                           checkpoint_dir=ckpt_dir, hooks=[]) as sess:
                        out_tensors = {name: sess.graph.get_tensor_by_name('{}:0'.format(name)) for name in out_node_names}
                        out_tensors.update(ids)
                        out_tensors.update({'global_step': global_step})

                        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

                        while not sess.should_stop() and not tf_feed.should_stop():

                            preds_dict = sess.run(out_tensors)
                            preds_dict.pop('global_step')

                            id_preds = np.concatenate([decode_str_fn(preds_dict[elem])
                                                       if elem[:-1] not in out_node_names else preds_dict[elem[:-1]] for
                                                       elem in parser.model_out_format], axis=1)
                            tf_feed.batch_results(id_preds)
                        if sess.should_stop():
                            tf_feed.terminate()


        pred_data = os.path.join(self.train_conf['pred_data'], self.args['date'])
        save_dir = os.path.join(self.train_conf['result_dir'], self.args['date'])
        if tf.gfile.Exists(save_dir):
            tf.gfile.DeleteRecursively(save_dir)
        predDataRDD = self.spark.read.text(Utils.get_files_under_dates(pred_data, self.parser.data_file_filter)).rdd.map(lambda x: x[0])
        cluster = TFCluster.run(sc, map_fun, tf_args, num_executors=num_executor, num_ps=num_ps, tensorboard=False,
                                input_mode=TFCluster.InputMode.SPARK)
        ResultRDD = cluster.inference(predDataRDD)
        ResultRDD.map(lambda x: delim.join(x)).saveAsTextFile(save_dir)
        cluster.shutdown()

class Utils(object):
    @staticmethod
    def get_files_under_dates(data_file, filter_func=None):
        _data_file, _args_date = data_file.rsplit('/', 1)
        if ':' in _args_date:
            data_file = _data_file
            startdate, enddate = _args_date.rsplit(':', 1)
            datelist = Utils._getDateInterval(startdate, enddate)
            # fileList = reduce(lambda x, y: x + y, [
            #     Utils._resc(os.path.join(data_file, date, file)) for date in datelist
            #     for file in tf.io.gfile.listdir(os.path.join(data_file, date))
            # ])
            fileList = reduce(lambda x, y: x + y, [
                Utils.Resc(os.path.join(data_file, date)) for date in datelist
            ])
        else:
            l = Utils.Resc(data_file)
            fileList = reduce(lambda x, y: x + y, [l])

        if filter_func is not None:
            fileList = list(filter(filter_func, fileList))
        return fileList
    @staticmethod
    def _resc(dict):
        if tf.gfile.IsDirectory(dict):
            out = []
            for fp, dirs, files in tf.gfile.Walk(dict):
                for dir in dirs:
                    out += Utils._resc(os.path.join(fp, dir))
                out += [os.path.join(fp, f) for f in files]
            return out
        else:
            return [dict]
    @staticmethod
    def Resc(data_file):
        try:
            files = Utils._resc(data_file)
            # files = [Utils._resc(os.path.join(data_file, file_dict)) for file_dict in tf.io.gfile.listdir(data_file)]
        except:
            if data_file[-1] != '/':
                data_file = data_file + '/'
            file_list_byte = subprocess.check_output(['aws', 's3', 'ls', data_file])
            items_under_data_file = file_list_byte.decode().split('\n')
            files = []
            for item in items_under_data_file:
                item = item.strip()
                if item == '':
                    pass
                elif item.startswith('PRE'):
                    # dir
                    pass
                else:
                    # file
                    file = item.rsplit(' ', 1)[1]
                    files.append(os.path.join(data_file, file))
        return files
    @staticmethod
    def _getDateInterval(sd, ed):
        interval = []
        sd = dt.datetime.strptime(sd, "%Y-%m-%d")
        ed = dt.datetime.strptime(ed, "%Y-%m-%d")
        while sd <= ed:
            interval.append(sd.strftime("%Y-%m-%d"))
            sd += dt.timedelta(days=1)
        return interval


def run(mode, conf, useSpark, retrain=False, date='', tw=1, wi=1, ps=1, logpath='', use_TFoS=False, num_executor=0, **kwargs):
    assert ps > 0
    logpath = os.getcwd() if logpath == '' else os.path.abspath(logpath)

    if not tf.gfile.Exists(logpath):
        tf.gfile.MakeDirs(logpath)
    if tf.gfile.Exists(os.path.join(logpath, conf)):
        conf = os.path.join(logpath, conf)
    open_log(logpath, mode)
    # _args = args
    if useSpark or mode in ['infer', 'test']:
        from pyspark.sql import SparkSession
        spark = SparkSession \
            .builder \
            .appName('dl_rank') \
            .getOrCreate()
    else:
        spark = None
    dataparser = conf_parser(conf, logpath, mode, useSpark=useSpark)
    print_config(dataparser.conf_dict)
    if retrain:
        model_path = dataparser.conf_dict['mission']['train']['model_dir']
        if tf.gfile.Exists(model_path):
            tf.gfile.DeleteRecursively(model_path)
    _args = {'mode': mode, 'date': date, 'tw': tw, 'wi': wi, 'ps': ps}
    eM = EstimatorManager(dataparser, _args, spark)
    eM.initModel(exPath=logpath)
    tf.compat.v1.logging.info("Build estimator: {}".format(eM.estimator))
    if mode == 'train':
        eM.train_and_eval(dataparser.parse_fn)
    elif mode == 'export':
        eM.export_model(**kwargs)
    elif mode == 'export_online':
        eM.export_model_online(**kwargs)
    elif mode == 'test':
        eM.args['mode'] = 'infer'
        return eM.test(dataparser.parse_fn)
    elif mode == 'infer':
        # setEnv()
        if use_TFoS:
            eM.predict_TFoS(dataparser.parse_fn, num_executor, ps, bake_path='hdfs:///user/hadoop/dl_rank')
        else:
            eM.predict(dataparser.parse_fn, bake_path=logpath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tw', type=int, help='number of machines', default=1)
    parser.add_argument('--wi', type=int, help='worker idx', default=1)
    parser.add_argument('--conf', type=str, help='switch I2Iconf_test')
    parser.add_argument('--date', type=str, help='date of data', default='')
    parser.add_argument('--mode', type=str, help='[train|infer|test|export|export_online]', default='train')
    parser.add_argument('--ps', type=int,
                        help='if True: use parameterServerStrategy else: use MultiWorkerMirroredStrategy', default=1)
    parser.add_argument('--tfconfig', type=str, help='json string of TF_CONFIG', default='')
    parser.add_argument('--useSpark', help='use spark-submit when infer', action='store_true')
    parser.add_argument('--logpath', type=str, help='path to write TF_CONFIG/train.log', default='')
    parser.add_argument('--retrain', help='if retrain, all summary will be clear',  action='store_true')
    parser.add_argument('--use_TFoS', help='path to write TF_CONFIG/train.log',  action='store_true')
    parser.add_argument('--num_executor', type=int, help='path to write TF_CONFIG/train.log',  default=0)
    args = parser.parse_args()
    if args.tfconfig != '':
        os.environ['TF_CONFIG'] = args.tfconfig.replace('_', '"').replace('*', ',')
        with open(os.path.join(args.logpath, 'tf_config.txt'), 'w') as f:
            f.write(os.environ['TF_CONFIG'])
    run(args.mode, args.conf, args.useSpark, args.retrain, args.date, args.tw, args.wi, args.ps, args.logpath,
        use_TFoS=args.use_TFoS, num_executor=args.num_executor)
