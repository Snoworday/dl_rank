import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict
import yaml

try:
    from dl_rank.layer import layers
except:
    from layer import layers

class BaseParser(object):
    def __init__(self, confPath, mode, useSpark):
        self.useSpark = useSpark
        self.confPath = confPath
        self.conf_dict = LoadDict(self.load_conf_switch(mode))
        self.column_to_csv_defaults()
        self.data_file_filter = lambda f_name: f_name.split('.')[-1] == 'txt'

    @property
    def model_out_format(self):
        if hasattr(self, '_model_out_format'):
            return self._model_out_format
        else:
            _model_out_format = self.conf_dict['separator']['model_out_format']
            self._model_out_format = [item+'_' for item in _model_out_format]
            return self._model_out_format

    def load_conf_switch(self, mode):
        confPath = self.confPath
        if self.useSpark:
            from pyspark import SparkFiles
            def wrapper(filename):
                try:
                    with open(SparkFiles.get(filename+'_'+mode+'.yaml'), 'r') as f:
                        return yaml.load(f)
                except:
                    with open(SparkFiles.get(filename+'.yaml'), 'r') as f:
                        return yaml.load(f)
            return wrapper
        else:
            def wrapper(filename):
                if tf.gfile.Exists(os.path.join(confPath, filename+'_'+mode+'.yaml')):
                    with tf.gfile.GFile(os.path.join(confPath, filename+'_'+mode+'.yaml'), 'r') as f:
                        return yaml.load(f)
                else:
                    with tf.gfile.GFile(os.path.join(confPath, filename+'.yaml'), 'r') as f:
                        return yaml.load(f)
            return wrapper

    def load_all_conf(self, *confs):
        all_conf = dict()
        for conf in confs:
            all_conf.update({conf: self.conf_dict[conf]})
        return all_conf

    def column_to_csv_defaults(self):
        """parse columns to record_defaults param in tf.decode_csv func
        Return:
            OrderedDict {'feature name': [''],...}
        """
        feature_conf = self.conf_dict['feature']
        feature_list_conf = self.conf_dict['schema']
        feature_list = [feature_list_conf[key] for key in sorted(feature_list_conf, reverse=False)]
        feature_unused = []
        csv_defaults = OrderedDict()
        csv_scope = OrderedDict()
        for f in feature_list:
            if f in feature_conf and not feature_conf[f]['ignore']:  # used features
                conf = feature_conf[f]
                scope = feature_conf[f]['parameter']['scope'] if 'scope' in feature_conf[f]['parameter'] else 'embedding'
                name = feature_conf[f]['parameter']['name'] if 'name' in feature_conf[f]['parameter'] else f
                csv_scope[f] = [scope, name]
                if conf['type'] == 'category':
                    csv_defaults[f] = ['']
                else:
                    csv_defaults[f] = ['']  # 0.0 for float32
            else:  # unused feature
                feature_unused.append(f)
                csv_defaults[f] = ['']
        self.feature_unused = feature_unused
        self.column_defaults = csv_defaults
        self.column_scope = csv_scope

    def serving_parse_fn(self, pred_node_names):
        csv_defaults = self.column_defaults
        csv_scope = self.column_scope
        feature_unused = self.feature_unused
        model_out_format = self.model_out_format
        def serving_input_receiver_fn():
            input_dict = dict()
            for key, value in csv_defaults.items():
                if key in feature_unused:
                    continue
                with tf.variable_scope('placeholder/'+csv_scope[key][0]) as scope:
                    if value[0] == '':
                        input_dict.update({key: tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name=csv_scope[key][1])})
                    else:
                        input_dict.update({key: tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name=csv_scope[key][1])})
            input_dict.update({id: tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name=id) for id in model_out_format if id[:-1] not in pred_node_names})
            return tf.estimator.export.ServingInputReceiver(features=input_dict, receiver_tensors=input_dict)
        return serving_input_receiver_fn


    def parse_fn(self, isPred=False, na_value='', tail=''):
        csv_defaults = self.column_defaults
        feature_unused = self.feature_unused
        primary_delim = self.conf_dict['separator']['primary_delim']
        secondary_delim = self.conf_dict['separator']['secondary_delim']
        train_data_format = self.conf_dict['separator']['train_data_format']
        infer_data_format = self.conf_dict['separator']['infer_data_format']
        def _parser_feature(data_container):
            data_container['features'] = tf.io.decode_csv(
                records=data_container['features'], record_defaults=list(csv_defaults.values()),
                field_delim=secondary_delim, use_quote_delim=False, na_value=na_value)
            features = dict(zip(csv_defaults.keys(), data_container['features']))
            for f in feature_unused:
                features.pop(f)
            features_tail = {key+tail: features[key] for key in features}
            return features_tail
        def parser(value):
            """Parse train and eval data with label
            Args:
                value: Tensor("arg0:0", shape=(), dtype=string)
            """
            if isPred:
                decode_data = tf.io.decode_csv(
                    records=value, record_defaults=[['']]*len(infer_data_format),
                    field_delim=primary_delim, use_quote_delim=False, na_value=na_value)
                data_container = dict(zip(infer_data_format, decode_data))
                features_tail = _parser_feature(data_container)
                features_tail.update({elem+'_'+tail: data for elem, data in data_container.items() if elem != 'features'})
                return features_tail
            else:
                decode_data = tf.io.decode_csv(
                    records=value, record_defaults=[['']]*len(train_data_format),
                    field_delim=primary_delim, use_quote_delim=False, na_value=na_value)
                data_container = dict(zip(train_data_format, decode_data))
                features_tail = _parser_feature(data_container)
                labels = [tf.equal(data_container[label], '1') for label in train_data_format if label != 'features']
                labels = tf.concat([tf.expand_dims(label, 1) if label.shape.ndims<2 else label for label in labels], axis=1)
                return features_tail, labels
        return parser

    def model_input_parse_fn(self, model_secondary_parse_fn):
        first_parser = self.first_parse_fn
        second_parser = self.secondary_parse_fn if hasattr(self, 'secondary_parse_fn') else model_secondary_parse_fn
        teriary_delim = self.conf_dict['separator']['teriary_delim']
        def wrapper(features, params, dims, model_struct):
            sparse_emb, deep_emb, dense_emb, mask = first_parser(features, params, teriary_delim)
            features = second_parser(sparse_emb, deep_emb, dense_emb, mask, model_struct)
            return dims, features
        return wrapper

    @staticmethod
    def first_parse_fn(features, params, teriary_delim):
        '''

        :param features:
        :param params:
        :param is_input_indices: if True: input feature has been changed to index, else string
        :return:
        sparse_features: [batch, one_hot_cate_con]
        deep_features: [batch, [cate, con], embedding_size]
        dense_features: [batch, con_num]
        '''
        def replace2default(tensor, intercept, default_value):
            if isinstance(tensor, tf.Tensor):
                isEffect = tf.cast(tf.less(tensor, intercept), tf.int32)
                tensor = isEffect * tensor + (1-isEffect) * default_value
                return tensor
            else:
                values = tensor.values
                isEffect = tf.cast(tf.less(values, intercept), tf.int32)
                values = isEffect * values + (1-isEffect) * default_value
                return tf.SparseTensor(indices=tensor.indices, values=values, dense_shape=tensor.dense_shape)

        def sparse_str2dense_num(f_input, field_num, type, null_value):
            null_value = str(null_value)
            f_input = layers.to_dense(f_input, depth=field_num, default_value=null_value)
            mask = tf.not_equal(f_input, null_value)
            f_output = tf.strings.to_number(f_input, out_type=type)
            return f_output, mask

        def sparse_str2sparse_num(f_input, field_num, type):
            values = f_input.values
            indices = f_input.indices
            dense_shape = f_input.dense_shape
            values = tf.strings.to_number(values, out_type=type)
            f_out = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
            sparse_f_indice = tf.SparseTensor(indices=indices,
                                              values=tf.squeeze(tf.split(indices, [1, 1], axis=1)[1], axis=1),
                                              dense_shape=dense_shape)
            return f_out, sparse_f_indice

        sparse = params['sparse']
        deep = params['deep']
        dense = params['dense']
        numeric = params['numeric']
        table = params['table']
        multi = params['multi']

        features = {key: tf.identity(features[key], name=key) for key in features}
        # batch_size = tf.shape(features[list(features.keys())[0]])[0]
        sparse_emb = OrderedDict()
        deep_emb = {'category': OrderedDict(), 'numeric': OrderedDict()} #cate, numeric
        dense_emb = OrderedDict()
        dense_emb_noreduce = OrderedDict()
        mask = dict()

        for f in features:
            num = multi[f][1]
            fill_value = multi[f][6]
            features_f = tf.strings.split(features[f], sep=teriary_delim, maxsplit=num)
            if fill_value == '':
                fill_mask = tf.squeeze(tf.where(tf.not_equal(features_f.values, '')), axis=1)
                features_f = tf.SparseTensor(indices=tf.gather(features_f.indices, fill_mask),
                                             values=tf.gather(features_f.values, fill_mask),
                                             dense_shape=features_f.dense_shape)
            else:
                features_f_val = tf.regex_replace(features_f.values, '^$', str(fill_value))
                features_f = tf.SparseTensor(indices=features_f.indices,
                                             values=features_f_val,
                                             dense_shape=features_f.dense_shape)
            features[f] = features_f

        for f in numeric:
            f_type, num, size, combiner, same, default_value, fill_value, null_value = multi[f]
            if combiner == 'none':
                features_f, f_mask = sparse_str2dense_num(features[f], abs(num), tf.float32,
                                                          null_value=null_value)
                dense_emb_noreduce.update({f: (features_f, f_mask)})
                if f in dense:
                    dense_emb.update({f: features_f})    #chi cun shi fou wei N, 1 haishi N,
                if num < 0 and f not in mask.keys():
                    mask[f] = f_mask
            else:
                sparse_f_float, sparse_f_indice = sparse_str2sparse_num(features[f], abs(num), tf.float32)
                # tfprint = tf.print('sparse_f_float', sparse_f_float, summarize=1e6)
                # with tf.control_dependencies([tfprint]):
                f_input_combine = layers.sparse_reduce(sparse_f_float, reduce_type=combiner)# tf.cast(tf.expand_dims(tf.nn.embedding_lookup_sparse(tf.ones([abs(num)], dtype=tf.float32), sparse_f_indice, sparse_f_float, combiner='sqrtn'), 1), dtype=tf.float32)
                # tp = tf.print('sparse_f_indice', sparse_f_indice, 'sparse_f_float', sparse_f_float, 'combiner', f_input_combine, 'thereis f_input_combine', features[f], f, summarize=1e6)
                # with tf.control_dependencies([tp]):
                #     f_input_combine = tf.identity(f_input_combine)
                dense_emb_noreduce.update({f: (sparse_f_float if not same else f_input_combine, sparse_f_indice)})
                if f in dense:
                    dense_emb.update({f: f_input_combine})

        for f in sparse:
            f_type, num, size, combiner, same, default_value, fill_value, null_value = multi[f]
            if combiner == 'none':
                if f in table.keys():
                    features_f_indice = table[f].lookup(features[f])
                    features_f_indicator = layers.to_dense(features_f_indice, depth=abs(num), default_value=int(null_value))
                    f_mask = tf.not_equal(features_f_indicator, null_value)
                else:
                    features_f_indicator, f_mask = sparse_str2dense_num(features[f], abs(num), tf.int32, null_value=null_value)
                    features_f_indicator = replace2default(features_f_indicator, size, default_value)
                # features_f_emb = tf.nn.embedding_lookup(sparse[f], features_f_indicator)
                features_f_emb = layers.multi_hot(features_f_indicator, num=abs(num), depth=size + 1 if null_value == -1 else size)
                if null_value == -1:
                    features_f_emb = features_f_emb[:, :, :-1]
                sparse_emb.update({f: features_f_emb})
                if num < 0 and f not in mask.keys():
                    mask[f] = f_mask
            else:
                if f in table.keys():
                    features_f_indice = table[f].lookup(features[f])
                else:
                    features_f_indice, _ = sparse_str2sparse_num(features[f], abs(num), tf.int32)
                    features_f_indice = replace2default(features_f_indice, size, default_value)
                # features_f_emb = tf.nn.embedding_lookup_sparse(sparse[f], features_f_indice, None, combiner=combiner)
                features_f_emb = layers.multi_hot(features_f_indice, depth=size, combiner=combiner)
                # _ = tf.reduce_mean(features_f_emb, axis=0, name='{}_embedding_params'.format(f))
                sparse_emb.update(
                    {f: tf.expand_dims(features_f_emb, 1)})

        for f in deep:
            f_type, num, size, combiner, same, default_value, fill_value, null_value = multi[f]
            if f_type == 'category':
                if combiner == 'none':
                    deep_f = tf.pad(deep[f], tf.constant([[0, 1], [0, 0]]), mode='CONSTANT',
                                    constant_values=0) if int(null_value) == size else deep[f]
                    if f in table.keys():
                        features_f_indice = table[f].lookup(features[f])
                        features_f_indicator = layers.to_dense(features_f_indice, depth=abs(num), default_value=int(null_value))
                        f_mask = tf.not_equal(features_f_indicator, int(null_value))
                    else:
                        # tfprint2 = tf.print(f, features[f], tf.shape(features[f]), 'start', summarize=10000)
                        # with tf.control_dependencies([]):
                        features_f_indicator, f_mask = sparse_str2dense_num(features[f], abs(num), tf.int32, null_value=null_value)
                        features_f_indicator = replace2default(features_f_indicator, size, default_value)
                    features_f_emb = tf.nn.embedding_lookup(deep_f, features_f_indicator)
                    deep_emb[f_type].update({f: features_f_emb})
                    if num < 0:
                        mask[f] = f_mask
                        _ = tf.identity(tf.multiply(tf.cast(tf.expand_dims(f_mask, -1), tf.float32), features_f_emb), name='{}_used_embedding_params'.format(f))    #bs,num.vec
                    else:
                        _ = tf.identity(features_f_emb, name='{}_used_embedding_params'.format(f))#bs,num,vec
                else:
                    if f in table.keys():
                        features_f_indice = table[f].lookup(features[f])
                    else:
                        features_f_indice, _ = sparse_str2sparse_num(features[f], abs(num), tf.int32)
                        features_f_indice = replace2default(features_f_indice, size, default_value)
                    if same:
                        features_f_emb = tf.tile(deep[f], tf.stack(tf.shape(features_f_indice.dense_shape)[0], 1))
                        assert False, 'u cant do this（set same in category feature） 0_0'
                    else:
                        features_f_emb = tf.nn.embedding_lookup_sparse(deep[f], features_f_indice, None, combiner=combiner)
                        features_f_emb = tf.pad(features_f_emb, [[0, tf.cast(features_f_indice.dense_shape[0], tf.int32)-tf.shape(features_f_emb)[0]], [0, 0]], 'CONSTANT', constant_values=0)
                        # tfprint3 = tf.print('features_f_emb_shape', features_f_indice.dense_shape, 'wait', tf.shape(features_f_emb), summarize=100000)
                        # with tf.control_dependencies([]):
                        features_f_emb = tf.identity(features_f_emb)
                    deep_emb[f_type].update(
                        {f: tf.expand_dims(features_f_emb, 1)})
                    _ = tf.identity(features_f_emb, name='{}_used_embedding_params'.format(f))
            elif f_type == 'numeric':
                features_f_value, sparse_f_indice = dense_emb_noreduce[f]
                if combiner == 'none':
                    features_f_emb = tf.multiply(tf.expand_dims(features_f_value, -1), tf.expand_dims(deep[f], 0))
                    deep_emb[f_type].update({f: features_f_emb})
                    sparse_f_indice = tf.cast(sparse_f_indice, tf.float32)
                    _ = tf.multiply(sparse_f_indice[:, :, np.newaxis], deep[f][np.newaxis, :, :], name='{}_used_embedding_params'.format(f))
                else:
                    if same:
                        features_f_combine = tf.expand_dims(tf.matmul(features_f_value, deep[f]), axis=1)
                        deep_emb[f_type].update({f: features_f_combine})
                        ###
                        _ = tf.identity(tf.expand_dims(deep[f], 0), name='{}_used_embedding_params'.format(f))#1,field_num,vec
                    else:
                        features_f_emb = layers.embedding_lookup_sparse(deep[f], sparse_f_indice, features_f_value, combiner=combiner)
                        _ = layers.embedding_lookup_sparse(deep[f], sparse_f_indice, None, combiner='mean', name='{}_used_embedding_params'.format(f))
                        # _ = tf.identity(features_f_emb, name='{}_used_embedding_params'.format(f))#bs,vec
                        features_f_combine = tf.expand_dims(features_f_emb, 1)
                        deep_emb[f_type].update({f: features_f_combine})

        deep_emb['category'].update(deep_emb['numeric'])
        deep_emb = deep_emb['category']
        return sparse_emb, deep_emb, dense_emb, mask

class LoadDict(dict):
    def __init__(self, load_fn):
        super(LoadDict, self).__init__()
        self.load_fn = load_fn

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            try:
                value = self.load_fn(item)
            except:
                value = None
            self.__setitem__(item, value)
            return value
