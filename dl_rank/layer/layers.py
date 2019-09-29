import tensorflow as tf


def multi_hot(cat_int_tensor, depth, num=0, combiner=None):
    if isinstance(cat_int_tensor, tf.Tensor):
        out = tf.concat([tf.expand_dims(tf.one_hot(cat_int_tensor[:, n], depth, axis=-1), 1) for n in range(num)], axis=1)
    else:
        indices = tf.concat([tf.expand_dims(cat_int_tensor.indices[:, 0], axis=1), tf.expand_dims(cat_int_tensor.values, axis=1)], axis=1)
        updates = tf.ones_like(cat_int_tensor.values)
        shape = [cat_int_tensor.dense_shape[0], depth]
        if combiner == 'sum':
            out = tf.scatter_nd(indices, updates, shape)
        elif combiner == 'mean':
            row_indices = indices[:, 0]
            row_sum = tf.scatter_nd(tf.expand_dims(row_indices, axis=1), updates, [tf.shape(cat_int_tensor)[0]])
            updates = tf.math.divide(1, tf.nn.embedding_lookup(row_sum, row_indices))
            out = tf.scatter_nd(indices, updates, shape)
        elif combiner == 'max':
            out = tf.scatter_nd(indices, updates, shape)
            out = tf.math.minimum(out, 1)
        else:
            assert False, "category features'one-hot embedding only support sum/mean/max "
        out = tf.cast(out, dtype=tf.float32)
    return out

def sparse_reduce(sparse_tensor, reduce_type):
    bs = tf.cast(sparse_tensor.dense_shape[0], dtype=tf.int32)
    if reduce_type=='sum':
        sparse_tensor_reduce = tf.sparse.reduce_sum(sparse_tensor, axis=1, keepdims=True)
    elif reduce_type=='max':
        sparse_tensor_reduce = tf.sparse.reduce_max(sparse_tensor, axis=1, keepdims=True)
    elif reduce_type=='mean':
        sparse_tensor_reduce = tf.sparse.reduce_sum(sparse_tensor, axis=1, keepdims=True)
        indices = tf.cast(sparse_tensor.indices, tf.int32)
        line_number_indices = indices[:, 0]
        line_count = tf.expand_dims(tf.math.bincount(line_number_indices, minlength=bs, dtype=tf.float32), 1)
        sparse_tensor_reduce = tf.div_no_nan(sparse_tensor_reduce, line_count)
    else:
        assert False, "no this way"
    sparse_tensor_scatter = tf.reshape(sparse_tensor_reduce, shape=[bs, 1])
    return sparse_tensor_scatter

def embedding_lookup_sparse(embedding_params, sparse_indices, weights, combiner, name=None):
    bs = tf.cast(sparse_indices.dense_shape[0], dtype=tf.int32)
    if combiner!='mean':
        embedding_tensor_reduce = tf.nn.embedding_lookup_sparse(embedding_params, sparse_indices, weights, combiner=combiner, name=name)
        embedding_tensor_scatter = tf.pad(embedding_tensor_reduce, [[0, bs-tf.shape(embedding_tensor_reduce)[0]], [0,0]], 'CONSTANT', constant_values=0)
    else:
        embedding_tensor_reduce = tf.nn.embedding_lookup_sparse(embedding_params, sparse_indices, weights, combiner='sum')
        embedding_tensor_scatter = tf.pad(embedding_tensor_reduce, [[0, bs-tf.shape(embedding_tensor_reduce)[0]], [0,0]], 'CONSTANT', constant_values=0)
        line_number = tf.cast(sparse_indices.indices[:, 0], tf.int32)
        line_count = tf.expand_dims(tf.math.bincount(line_number, minlength=bs, dtype=tf.float32), 1)
        embedding_tensor_scatter = tf.div_no_nan(embedding_tensor_scatter, line_count, name=name)
    return embedding_tensor_scatter

def to_dense(tensor, depth, default_value=0):
    bs = tf.cast(tensor.dense_shape[0], tf.int32)
    tensor_dense = tf.sparse.to_dense(tensor, default_value=default_value)
    tfprint = tf.print('dense_shape', tf.shape(tensor_dense), 'tensor_dense', tensor_dense,  'tensor', tensor, 'tensor shape', tf.shape(tensor), summarize=100000)
    # with tf.control_dependencies([]):
    tensor_pad = tf.pad(tensor_dense, [[0, bs-tf.shape(tensor_dense)[0]], [0,depth-tf.shape(tensor_dense)[1]]], 'CONSTANT', constant_values=default_value)
    tensor_reshape = tf.reshape(tensor_pad, [-1, depth])
    return tensor_reshape

def _init_weights_bias(input_dim, out_dim):
    weights = tf.Variable(
        tf.random.normal([input_dim, out_dim], 0.0, 0.01)
    )
    bias = tf.Variable(
        tf.random.normal([out_dim], 0.0, 0.01)
    )
    return weights, bias

def dense(input, out_dim, activation=None, bn=False, training=True):
    # w_out, b_out = self._init_weights_bias(int(input.shape[1]), out_dim)
    # output = tf.compat.v1.nn.dense(input, w_out, b_out)
    output = tf.layers.dense(input, out_dim, trainable=training)
    if bn:
        output = tf.compat.v1.layers.batch_normalization(output, training=training)
    if activation is not None:
        output = activation(output)
    return output

def focal_loss_sigmoid(labels,logits,weights=1,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    ratio = (tf.cast(tf.shape(labels)[0], dtype=tf.float32)) / (tf.reduce_sum(weights) + 1)
    L = L * weights
    L = tf.reduce_mean(L) * ratio
    return L

def l1_loss(l):
    pass

def l2_loss(l):
    pass

def MBA_loss(l):
    nodes = {k: v for k, v in tf.get_default_graph()._nodes_by_name.items() if 'used_embedding_params' in k}
    mini_batch_aware_reg = 0
    for _, node in nodes.items():
        try:
            mini_batch_aware_reg += tf.reduce_sum(tf.reduce_mean(tf.square(node.outputs), axis=0))
        except:
            a=1
    return mini_batch_aware_reg * l
