import tensorflow as tf
# from dl_rank.model import baseModel
from .BaseModel import baseModel
try:
    from dl_rank.layer import layers
except:
    from layer import layers

class deepfm(baseModel):
    def __init__(self, model_conf, mode):
        super(deepfm, self).__init__(model_conf, mode)
        self.dropout_keep_fm = model_conf['dropout_keep_fm']
        self.dropout_keep_deep = model_conf['dropout_keep_deep']

        self.use_first_order = self.model_conf['first_order']
        self.dnn_hidden_units = self.model_conf['dnn_hidden_units']

    def _forward_model(self, is_training, *args, **kwargs):
        sparse_features, deep_features, dense_features = args
        with tf.compat.v1.variable_scope('trunk'):
            if self.use_first_order:
                deep_features, bias_features = tf.split(deep_features, [-1, 1], axis=2)
                bias_features = tf.squeeze(bias_features, axis=2)
                cate_bias_features, con_bias_features = tf.split(bias_features, [self.dims['cate_num'], self.dims['con_num']],
                                                                 axis=1)
            cate_deep_features, con_deep_features = tf.split(deep_features,
                                                             [self.dims['deep_num'] - self.dims['con_deep_num'],
                                                              self.dims['con_deep_num']], axis=1)
            with tf.compat.v1.variable_scope('fm'):
                if self.use_first_order:
                    y_first_order = tf.concat([cate_bias_features, con_bias_features], axis=1)
                    y_first_order = tf.compat.v1.layers.batch_normalization(y_first_order, training=is_training)
                    y_first_order = tf.compat.v1.layers.dropout(y_first_order, 1-self.dropout_keep_fm, training=is_training)

                fm_features = tf.concat([con_deep_features, cate_deep_features], axis=1)
                square_sum = tf.reduce_sum(input_tensor=tf.square(fm_features), axis=1)
                sum_square = tf.square(tf.reduce_sum(input_tensor=fm_features, axis=1))
                y_second_order = 0.5 * tf.subtract(sum_square, square_sum)
                y_second_order = tf.compat.v1.layers.batch_normalization(y_second_order, training=is_training)
                y_second_order = tf.compat.v1.layers.dropout(y_second_order, 1-self.dropout_keep_fm, training=is_training)

            with tf.compat.v1.variable_scope('dnn'):
                deep_input = tf.concat([tf.reshape(cate_deep_features, [-1, (self.dims['deep_num']-self.dims['con_deep_num'])*self.dims['s_embed_size']]), dense_features], axis=1)
                deep_input = tf.compat.v1.layers.dropout(deep_input, 1 - self.dropout_keep_deep, training=is_training)
                for dim in self.dnn_hidden_units:
                    deep_input = layers.dense(deep_input, dim, tf.nn.relu, bn=True, training=is_training)
                    deep_input = tf.compat.v1.layers.dropout(deep_input, 1 - self.dropout_keep_deep, training=is_training)
            if self.use_first_order:
                merge = tf.concat([y_first_order, y_second_order, deep_input], axis=1)
            else:
                merge = tf.concat([y_second_order, deep_input], axis=1)
            out = layers.dense(merge, 1, bn=False, training=is_training)
        predictions = tf.nn.sigmoid(out, name=self.out_node_names[0])
        return predictions


    def get_loss(self, labels, predictions):
        labels = tf.cast(labels, tf.int32)
        l1_reg, l2_reg = self.model_conf['l1_reg'], self.model_conf['l2_reg']
        loss = tf.compat.v1.losses.log_loss(labels, predictions)
        T_vars = tf.compat.v1.trainable_variables()
        M_vars = [var for var in T_vars if var.name.startswith('trunk')]
        E_vars = list(set(T_vars)-set(M_vars))
        l1_loss, l2_loss = 0, 0
        if l2_reg > 0:
            for vars in E_vars:
                l2_loss += tf.keras.regularizers.l2(
                    0.5 * (l2_reg))(vars)
        if l1_reg > 0:
            for vars in M_vars:
                l1_loss += tf.keras.regularizers.l1(
                    l1_reg)(vars)
        loss = loss + l1_loss + l2_loss
        return loss


    # def get_precision(self, labels, predictions):
    #     labels = tf.cast(labels, tf.int32)
    #     labels = tf.expand_dims(labels, axis=1)
    #
    #     return tf.reduce_mean(
    #         input_tensor=tf.cast(
    #             tf.less(tf.abs(predictions - tf.cast(labels, tf.float32)), 0.5),
    #             tf.float32))

    def get_eval_metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        labels = tf.cast(labels, tf.int32)
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
        return predictions_out
