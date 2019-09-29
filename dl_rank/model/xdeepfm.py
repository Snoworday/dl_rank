import tensorflow as tf
from .BaseModel import baseModel
try:
    from dl_rank.layer import layers
except:
    from layer import layers

class xdeepfm(baseModel):
    name = 'xdeepfm'
    def forward(self, wide_features, deep_features, dense_features, deep_embedding_columns, load_conf, is_training):
        with tf.variable_scope('xdeepfm'):
            deep_input = tf.split(deep_features, num_or_size_splits=self.model_conf['field_num'], axis=1)
            deep_input = tf.concat([tf.expand_dims(feature, axis=1) for feature in deep_input], axis=1)
            cross_layer_size = self.model_conf['cross_layer_size']
            final_result = []
            final_len = 0
            embed_dim = self.model_conf['embed_dim']
            field_nums = [self.model_conf['field_num']]
            cin_layers = [deep_input]
            for idx, layer_size in enumerate(cross_layer_size):
                dot_result = tf.einsum('ipk,iqk->ipqk', cin_layers[0], cin_layers[-1])
                dot_result = tf.reshape(dot_result, shape=[-1, embed_dim, field_nums[0] * field_nums[-1]])
                filters = tf.get_variable(initializer=self.initializer_fn, name='f_' + str(idx),
                                          shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                          dtype=tf.float32)

                # [batch_size, embed_dim, layers_size]
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                final_len += layer_size
                field_nums.append(layer_size)
                final_result.append(curr_out)
                curr_out = tf.transpose(curr_out, [0, 2, 1])
                cin_layers.append(curr_out)
            result = tf.concat(final_result, axis=2)
            result = tf.reduce_sum(result, axis=1)  # [batch_size, layer_size]
            y_xdeepfm = tf.layers.dense(result, 1, activation=None,
                                        kernel_initializer=self.initializer_fn)
        with tf.variable_scope('dnn'):
            dnn_hidden_units = self.model_conf['dnn_hidden_units']
            deep_input = tf.concat([deep_features, dense_features], axis=1)
            for dim in dnn_hidden_units:
                deep_input = tf.layers.dense(deep_input, dim, activation=self.activation_fn,
                                             kernel_initializer=self.initializer_fn)
            y_dnn = tf.layers.dense(deep_input, 1, activation=None,
                                    kernel_initializer=self.initializer_fn)

        return y_xdeepfm + y_dnn + tf.reduce_sum(wide_features, axis=1, keep_dims=True)