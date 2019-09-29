import tensorflow as tf
from .BaseModel import baseModel
try:
    from dl_rank.layer import layers
except:
    from layer import layers


# wide columns

class deepfm(baseModel):
    name = 'deepfm_v2'
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
            ctr_out = tf.nn.sigmoid(layers.dense(merge, 1, bn=False, training=is_training))
            cvr_out = tf.nn.sigmoid(layers.dense(merge, 1, bn=False, training=is_training))
            ctcvr_out = ctr_out * cvr_out
            out = tf.concat([ctr_out, ctcvr_out], axis=1)
        predictions = tf.identity(out, name=self.out_node_names[0])
        return predictions

    def get_eval_metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        ctr_labels, uv_labels = tf.split(tf.cast(labels, tf.int32), [1, 1], axis=1)
        ctr_preds, uv_preds = tf.split(predictions, [1, 1], axis=1)
        ctr_auc = tf.compat.v1.metrics.auc(
            labels=ctr_labels,
            predictions=ctr_preds,
            name='auc',
            curve='ROC',
        )

        uv_auc = tf.compat.v1.metrics.auc(
            labels=uv_labels,
            predictions=uv_preds,
            name='auc',
            curve='ROC',
        )
        return {
            'ctr_auc': ctr_auc, 'uv_auc': uv_auc
        }

    def get_loss(self, labels, predictions):
        ctr_labels, uv_labels = tf.split(tf.cast(labels, tf.int32), [1, 1], axis=1)
        ctr_pred, uv_pred = tf.split(predictions, [1, 1], axis=1)
        ctr_labels_ = tf.cast(ctr_labels, tf.float32)

        l1_reg, l2_reg, MBA_reg = self.model_conf['l1_reg'], self.model_conf['l2_reg'], self.model_conf['MBA_reg']
        ctr_loss = tf.compat.v1.losses.log_loss(ctr_labels, ctr_pred)
        uv_loss = layers.focal_loss_sigmoid(uv_labels, uv_pred, weights=ctr_labels_, alpha=0.03)
        T_vars = tf.compat.v1.trainable_variables()
        M_vars = [var for var in T_vars if var.name.startswith('trunk')]
        E_vars = list(set(T_vars)-set(M_vars))

        l1_loss, l2_loss = 0, 0
        if l2_reg > 0:
            for vars in M_vars:
                l2_loss += tf.keras.regularizers.l2(
                    l2_reg)(vars)
        if l1_reg > 0:
            for vars in M_vars:
                l1_loss += tf.keras.regularizers.l1(
                    l1_reg)(vars)
        loss = ctr_loss + uv_loss + l1_loss + l2_loss + layers.MBA_loss(MBA_reg) # + dice_loss * l2_reg
        return loss

    def add_summary(self, labels, predictions, eval_metric_ops):
        ctr_labels, uv_labels = tf.split(tf.cast(labels, tf.int32), [1, 1], axis=1)
        ctr_pred, uv_pred = tf.split(predictions, [1, 1], axis=1)

        ctr_precision = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.less(tf.abs(ctr_pred - tf.cast(ctr_labels, tf.float32)), 0.5),
                tf.float32))
        uv_precision = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.less(tf.abs(uv_pred - tf.cast(uv_labels, tf.float32)), 0.5),
                tf.float32))

        tf.compat.v1.summary.scalar('uv_precision', uv_precision)
        tf.compat.v1.summary.scalar('ctr_precision', ctr_precision)
        tf.compat.v1.summary.scalar('ctr_auc_train', eval_metric_ops['ctr_auc'][1])
        tf.compat.v1.summary.scalar('uv_auc_train', eval_metric_ops['uv_auc'][1])
