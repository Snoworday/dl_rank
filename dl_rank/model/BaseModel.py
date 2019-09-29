from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from functools import reduce
import numpy as np
import warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
if tf.__version__[0]=='1':
    lookup = tf.contrib.lookup
else:
    lookup = tf.lookup
    assert False, 'emm.., lookup.index_table_from_tensor has been removed in TF2 ... and no substitute Q_Q'

class baseModel(metaclass=ABCMeta):
    name = 'base'
    def __init__(self, model_conf, mode):
        # self.activation_fn = activation_fn
        # self.initializer_fn = initializer_fn
        self.model_conf = model_conf
        self.mode = mode
        self.num_shards = self.model_conf['num_shards'] if 'num_shards' in self.model_conf else 1
        self.placeholder_map = self.model_conf['input_node_map'] if 'input_node_map' in self.model_conf else dict()
        self.out_node_names = self.model_conf['out_node_names'] if 'out_node_names' in self.model_conf else ['out']
    def set_embedding_parser(self, fn):
        self.embedding_parser_fn = fn(self._secondary_parse_fn)

    @staticmethod
    def build_embedding(params, num_shards):
        feature_conf = params['feature_conf']
        feature_list_conf = params['feature_list']
        feature_list = [feature_list_conf[key] for key in sorted(feature_list_conf, reverse=False)]
        model_conf = params['model_conf']
        vocabulary_conf = params['vocabulary_conf']
        embed_dim = model_conf['embed_dim']
        first_order = int(model_conf['first_order'])
        partitioner = tf.fixed_size_partitioner(num_shards) if num_shards > 1 else None

        table = OrderedDict()
        sparse = []
        deep = OrderedDict()
        multi = OrderedDict()
        model_struct = defaultdict(list)
        numeric = []
        dense = []
        dense_tag = []
        wide_dim, deep_dim, deep_num, cate_num, con_num, all_num, con_deep_num, con_bias_num = 0, 0, 0, 0, 0, 0, 0, 0
        for feature in feature_list:
            if not feature in feature_conf:
                continue
            conf = feature_conf[feature]
            if conf['ignore']:
                continue
            f_type, f_tran, f_param = conf['type'], conf['transform'], conf['parameter']
            if 'group' in conf:
                for struct in conf['group']:
                    model_struct[struct].append(feature)
            f_multi = conf['multi'] if 'multi' in conf else {'num': 1, 'same': True, 'combiner': 'none'}
            feature_name = f_param['name'] if 'name' in f_param else feature
            feature_embed_dim = f_param['embed_dim'] if 'embed_dim' in f_param else embed_dim
            feature_scope = f_param['scope'] if 'scope' in f_param else 'embedding'

            with tf.variable_scope(feature_scope, reuse=tf.AUTO_REUSE, partitioner=partitioner) as scope:
                if f_type == 'category':
                    f_num, combiner = f_multi['num'], f_multi['combiner']
                    default_value = f_param['default'] if 'default' in f_param else 0
                    if combiner != 'none' and f_num >= 1:
                        f_num = 1
                    if f_tran == 'vocabulary_list':
                        vocabulary = vocabulary_conf[feature]
                        vocabulary = ['DEFAULT'] + vocabulary
                        table.update(
                            {feature: lookup.index_table_from_tensor(mapping=tf.constant(vocabulary), default_value=default_value)})
                        f_dim = len(vocabulary)*f_num
                        f_size = len(vocabulary)
                        fill_value = f_param['fill'] if 'fill' in f_param else ''#'DEFAULT'
                    elif f_tran == 'tabled':
                        f_dim = f_param['size']*f_num
                        f_size = f_param['size']
                        fill_value = f_param['fill'] if 'fill' in f_param else ''#'0'
                    else:
                        assert False, 'only support category features with vocabulary or tabled'

                    if 'onehot' in conf['style']:
                        sparse.append(feature)
                        if f_num >= 1:
                            wide_dim += f_dim * f_num
                        else:
                            wide_dim += f_dim * (-f_num)
                    if 'embedding' in conf['style']:
                        deep.update({feature: tf.get_variable(initializer=tf.random.normal([f_size, feature_embed_dim+first_order], 0.0, 0.1),
                                                          name='{}_embedding'.format(feature_name))})

                        if f_num >= 1:
                            deep_dim += (feature_embed_dim+first_order) * f_num
                            deep_num += f_num
                        else:
                            deep_dim += (feature_embed_dim + first_order) * (-f_num)
                            deep_num += -f_num

                    f_multi['same'] = False
                    dense_tag += [0] * abs(f_num)
                    cate_num += abs(f_num)
                    all_num += abs(f_num)
                    tail_value = f_param['tail'] if 'tail' in f_param else f_size

                elif f_type == 'numeric':
                    f_size = 1
                    f_num = f_multi['num']
                    numeric.append(feature)
                    if 'value' in conf['style']:
                        dense.append(feature)
                    if 'embedding' in conf['style']:
                        if f_num >=1:
                            if f_multi['combiner'] == 'none':
                                if f_multi['same']:
                                    deep.update({feature:
                                        tf.get_variable(
                                            initializer=tf.random.normal([1, embed_dim + first_order], 0.0,
                                                                         0.1),
                                            name='{}'.format(feature_name))})
                                else:
                                    deep.update({feature: tf.get_variable(
                                        initializer=tf.random.normal([f_num, embed_dim + first_order], 0.0, 0.1),
                                        name='{}'.format(feature_name))})
                                con_num += f_num
                                all_num += f_num
                                con_deep_num += f_num
                                deep_num += f_num
                                deep_dim += (embed_dim+first_order) * f_num
                                dense_tag += [1]*f_num
                            else:
                                con_num += 1
                                all_num += 1
                                if f_multi['same']:
                                    deep.update({feature: tf.get_variable(
                                        initializer=tf.random.normal([1, embed_dim + first_order], 0.0, 0.1),
                                        name='{}'.format(feature_name))})
                                else:
                                    deep.update({feature: tf.get_variable(
                                        initializer=tf.random.normal([f_num, embed_dim + first_order], 0.0, 0.1),
                                        name='{}'.format(feature_name))})
                                con_deep_num += 1
                                deep_num += 1
                                deep_dim += embed_dim+first_order
                                dense_tag.append(1)

                        else:
                            if f_multi['combiner'] == 'none':
                                f_num = -f_num
                                if f_multi['same']:
                                    deep.update({feature:
                                                     tf.get_variable(
                                                         initializer=tf.random.normal([1, embed_dim + first_order], 0.0,
                                                                                      0.1),
                                                         name='{}_embedding'.format(feature_name))})
                                else:
                                    deep.update({feature: tf.get_variable(initializer=tf.random.normal([f_num, embed_dim+first_order], 0.0, 0.1), name='{}_embedding'.format(feature))})
                                con_num += f_num
                                all_num += f_num
                                con_deep_num += f_num
                                deep_num += f_num
                                deep_dim += (embed_dim+first_order) * f_num
                                dense_tag += [1]*f_num
                            else:
                                f_num -= f_num
                                con_num += 1
                                all_num += 1
                                if f_multi['same']:
                                    deep.update({feature: tf.get_variable(initializer=tf.random.normal([1, embed_dim+first_order], 0.0, 0.1), name='{}_embedding'.format(feature))})
                                else:
                                    deep.update({feature: tf.get_variable(initializer=tf.random.normal([f_num, embed_dim+first_order], 0.0, 0.1), name='{}_embedding'.format(feature))})
                                con_deep_num += 1
                                deep_num += 1
                                deep_dim += embed_dim+first_order
                                dense_tag.append(1)

                    f_dim = -1
                    default_value = 0
                    fill_value = f_param['fill'] if 'fill' in f_param else ''#'0'
                    tail_value = f_param['tail'] if 'tail' in f_param else 0
                else:
                    assert False, "cant't handle this type now: {}".format(f_type)
                multi.update(
                    {feature: (f_type, f_multi['num'], f_size, f_multi['combiner'], f_multi['same'], default_value, fill_value, tail_value)}
                )

        dims = {'deep_num': deep_num, 'deep_dim': deep_dim, 'wide_dim': wide_dim, 'con_num': con_num, 'cate_num': cate_num,
                's_embed_size': embed_dim, 'cate_deep_num': deep_num-con_deep_num,
                'd_embed_size': embed_dim, 'all_num': all_num, 'dense_tag': dense_tag, 'con_deep_num': con_deep_num}
        columns = {'table': table, 'sparse': sparse, 'deep': deep, 'dense': dense, 'numeric': numeric, 'dense_tag': dense_tag, 'multi': multi}
        # dense_tag = tf.constant(dense_tag)
        return model_struct, columns, dims

    @staticmethod
    def _secondary_parse_fn(sparse_emb, deep_emb, dense_emb, mask, model_struct):
        sparse_features = tf.concat([tf.reshape(embeddings, [-1, reduce(lambda x, y:x*y, embeddings.get_shape().as_list()[1:])]) for name, embeddings in sparse_emb.items()], axis=1)
        deep_features = tf.concat([embeddings for name, embeddings in deep_emb.items()], axis=1)
        dense_features = tf.concat([embeddings for name, embeddings in dense_emb.items()], axis=1)
        return sparse_features, deep_features, dense_features

    def build_columns_information(self, params):
        self.model_struct, *Features = baseModel.build_embedding(params, self.num_shards)
        return Features

    def get_train_op_fn(self, loss, params):
        global_step = tf.compat.v1.train.get_global_step()
        learning_rate = tf.compat.v1.train.exponential_decay(self.model_conf['learning_rate'], global_step,
                                                             self.model_conf['lr_decay_step'], 0.96, staircase=True)
        if self.model_conf['optimizer'].lower() == 'adadelta':
            optimizer = tf.compat.v1.train.AdadeltaOptimizer
        elif self.model_conf['optimizer'].lower() == 'sgd':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer
        elif self.model_conf['optimizer'].lower() == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer
        elif self.model_conf['optimizer'].lower() == 'rmsprop':
            optimizer = tf.compat.v1.train.RMSPropOptimizer
        elif self.model_conf['optimizer'].lower() == 'ftrl':
            optimizer = tf.compat.v1.train.FtrlOptimizer
        else:
            assert False, '??? {} is what optimizer?'.format(self.model_conf['optimizer'])

        train_op = optimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(loss, global_step=global_step)
        return train_op

    def get_loss(self, labels, predictions):
        labels = tf.cast(labels, tf.int32)
        # labels = tf.expand_dims(labels, axis=1)
        l1_reg, l2_reg = self.model_conf['l1_reg'], self.model_conf['l2_reg']
        loss = tf.compat.v1.losses.log_loss(labels, predictions)
        T_vars = tf.compat.v1.trainable_variables()
        M_vars = [var for var in T_vars if var.name.startswith('trunk')]
        E_vars = list(set(T_vars)-set(M_vars))
        l1_loss, l2_loss = 0, 0
        if l2_reg > 0:
            for vars in T_vars:
                l2_loss += tf.keras.regularizers.l2(
                    0.5 * (l2_reg))(vars)
        if l1_reg > 0:
            for vars in M_vars:
                l1_loss += tf.keras.regularizers.l1(
                    l1_reg)(vars)
        loss = loss + l1_loss + l2_loss
        return loss

    @abstractmethod
    def get_eval_metric_ops(self, labels, predictions):
        pass

    def forward(self, indexed_input, params, is_training):
        columns, dims = self.build_columns_information(params)
        self.dims, model_input = self.embedding_parser_fn(indexed_input, columns, dims, self.model_struct)
        out = self._forward_model(is_training, *model_input)
        if self.mode != 'train':
            for out_node in self.out_node_names:
                try:
                    _ = tf.get_default_graph().get_operation_by_name(out_node)
                except:
                    raise LookupError('Cant find out node:{} in Graph'.format(out_node))
        return out

    @abstractmethod
    def _forward_model(self, is_training, *args, **kwargs):
        '''
        :param sparse_features: [batch_size, len of all one hot merge], one_hot, only from category field generally
        :param deep_features: [batch_size, [cate, con]num, embedding_dim], embedding, concat of category field and numeric
        :param dense_features: [batch_size, len of con], concat real value of numeric data
        :param is_training: bool
        :return:
        '''
        pass

    def add_summary(self, labels, predictions, eval_metric_ops):
        labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=1)
        precision = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.less(tf.abs(predictions - labels), 0.5),
                tf.float32))
        tf.compat.v1.summary.scalar('precision', precision)
        tf.compat.v1.summary.scalar('auc_train', eval_metric_ops['auc'][1])

    def get_predictions_out(self, features, predictions, pids, out_format):
        predictions_out = {id:tf.expand_dims(tf.identity(pids[id], name=id), axis=1) if id[:-1] not in self.out_node_names else predictions
                           for id in out_format}
        return predictions_out
