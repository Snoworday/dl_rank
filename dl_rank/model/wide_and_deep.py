import tensorflow as tf
from .BaseModel import baseModel
import sys
sys.path.append("..")
try:
    from dl_rank.layer import layers
except:
    from layer import layers

class wdl(baseModel):
    name = 'wdl'
    def __init__(self, model_conf, mode):
        super(wdl, self).__init__(model_conf, mode)
        self.dropout_keep_fm = model_conf['dropout_keep_fm']
        self.dropout_keep_deep = model_conf['dropout_keep_deep']
        self.dnn_hidden_units = self.model_conf['dnn_hidden_units']
        self.use_first_order = self.model_conf['first_order']

    def _forward_model(self, is_training, *args, **kwargs):
        sparse_features, deep_features, dense_features = args
        with tf.compat.v1.variable_scope('trunk'):
            if self.use_first_order:
                deep_features, bias_features = tf.split(deep_features, [-1, 1], axis=2)
            cate_deep_features, con_deep_features = tf.split(deep_features,
                                                             [self.dims['deep_num'] - self.dims['con_deep_num'],
                                                              self.dims['con_deep_num']], axis=1)
            with tf.compat.v1.variable_scope('wide'):
                wide_out = layers.dense(sparse_features, 1)
            with tf.compat.v1.variable_scope('deep'):
                deep_input = tf.concat([tf.reshape(cate_deep_features, [-1, (self.dims['deep_num']-self.dims['con_deep_num'])*self.dims['s_embed_size']]), dense_features], axis=1)
                deep_input = tf.compat.v1.layers.dropout(deep_input, 1 - self.dropout_keep_deep, training=is_training)
                for dim in self.dnn_hidden_units:
                    deep_input = layers.dense(deep_input, dim, bn=True, activation=tf.nn.relu, training=is_training)
                    deep_input = tf.compat.v1.layers.dropout(deep_input, 1 - self.dropout_keep_deep, training=is_training)
                deep_out = layers.dense(deep_input, 1)
        out = wide_out + deep_out
        predictions = tf.nn.sigmoid(out, name=self.out_node_names[0])
        ld = wide_out
        return predictions

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

    def get_predictions_out(self, features, predictions, pids, out_format):
        predictions_out = {id:tf.expand_dims(tf.identity(pids[id], name=id), axis=1) if id != 'out' else predictions
                           for id in out_format}
        # predictions_out.update({'pred': predictions})
        # pid1 = tf.expand_dims(tf.identity(pids['pid1'], name='pid1'), axis=1)
        # pid2 = tf.expand_dims(tf.identity(pids['pid2'], name='pid2'), axis=1)
        # tf.compat.v1.logging.info('keyss:{}'.format(features.keys()))
        # predictions_out = {'pid1': pid1, 'pid2': pid2, 'prob': predictions}
        return predictions_out
