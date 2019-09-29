import tensorflow as tf
from .BaseModel import baseModel
try:
    from dl_rank.layer import layers
except:
    from layer import layers

class dcn(baseModel):
    name = 'dcn'
    def __init__(self, model_conf, mode):
        super(dcn, self).__init__(model_conf, mode)
        self.dropout_keep_deep = model_conf['dropout_keep_deep']

        self.use_first_order = self.model_conf['first_order']
        self.dnn_hidden_units = self.model_conf['dnn_hidden_units']
        self.cross_dcn_layer_size = self.model_conf['cross_dcn_layer_size']

    def _forward_model(self, is_training, *args, **kwargs):
        sparse_features, deep_features, dense_features = args
        with tf.compat.v1.variable_scope('trunk'):
            if self.use_first_order:
                deep_features, bias_features = tf.split(deep_features, [-1, 1], axis=2)
            cate_deep_features, con_deep_features = tf.split(deep_features,
                                                             [self.dims['deep_num'] - self.dims['con_deep_num'],
                                                              self.dims['con_deep_num']], axis=1)
            x0 = tf.concat([dense_features, tf.reshape(cate_deep_features, [-1, (self.dims['deep_num']-self.dims['con_deep_num'])*self.dims['s_embed_size']])], axis=1)
            x0_ = tf.expand_dims(x0, axis=2)
            with tf.compat.v1.variable_scope('cross'):
                cross = x0_
                for i in range(self.cross_dcn_layer_size):
                    cross = layers.dense(tf.compat.v1.matmul(x0_, cross, transpose_b=True), 1, bn=True, training=is_training) + x0_
                cross = tf.squeeze(cross, axis=2)
            with tf.compat.v1.variable_scope('deep'):
                deep_input = x0
                for dim in self.dnn_hidden_units:
                    deep_input = layers.dense(deep_input, dim, tf.nn.relu, bn=True, training=is_training)
                    deep_input = tf.compat.v1.layers.dropout(deep_input, 1 - self.dropout_keep_deep, training=is_training)
                deep_out = deep_input
            merge = tf.concat([cross, deep_out], axis=1)
            out = layers.dense(merge, 1, bn=False, training=is_training)
            predictions = tf.nn.sigmoid(out, name=self.out_node_names[0])
        ld = deep_out
        return predictions

    def get_eval_metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        labels = tf.expand_dims(tf.cast(labels, tf.int32), axis=1)
        auc = tf.compat.v1.metrics.auc(
            labels=labels,
            predictions=predictions,
            name='auc',
            curve='ROC',
        )

        return {
            'auc': auc
        }

    def get_predictions_out(self, features, predictions, pids):
        pid1 = tf.expand_dims(tf.identity(pids['pid1'], name='pid1'), axis=1)
        pid2 = tf.expand_dims(tf.identity(pids['pid2'], name='pid2'), axis=1)
        tf.compat.v1.logging.info('keyss:{}'.format(features.keys()))
        predictions_out = {'pid1': pid1, 'pid2': pid2, 'prob': predictions}
        return predictions_out