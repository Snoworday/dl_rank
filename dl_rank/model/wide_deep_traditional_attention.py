import tensorflow as tf
from .BaseModel import baseModel
from functools import reduce
from operator import mul
try:
    from dl_rank.layer import layers
except:
    from layer import layers

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


# wide columns

class _config():
    def __init__(self, cfg):
        self.reduce_sequence_length = cfg['reduce_sequence_length']
        self.embedding_dim = cfg['reduce_sequence_length']
        self.dropout = cfg['dropout']
        self.last_dropout = cfg['last_dropout']
        self.hidden_num = cfg['hidden_num']
        self.wd = cfg['wd']
        self.out_node_name = ['predict_value']



class wide_and_deep(baseModel):
    name = 'wide_deep_traditional_attention'
    def __init__(self, model_conf, mode):
        super(wide_and_deep, self).__init__(model_conf, mode)
        self.dropout_keep_fm = model_conf['dropout_keep_fm']
        self.dropout_keep_deep = model_conf['dropout_keep_deep']
        self.optimizer_name = model_conf['optimizer']
        self.l1 = model_conf['l1_reg']
        self.l2 = model_conf['l2_reg']
        self.lr = model_conf['learning_rate']

        # self.use_first_order = self.model_conf['first_order']
        self.dnn_hidden_units = self.model_conf['dnn_hidden_units']

        self.batch_size = 128
        self.embedding_size = 16
        self.cfg = _config(model_conf['cfg'])
        self.mode = mode

    def _forward_model(self, is_training, *args, **kwargs):
        # sparse_features, deep_features, dense_features = _args
        self.is_train = is_training
        with tf.variable_scope('wide_layer_variable') as scope:
            self.b = tf.get_variable('W', dtype=tf.float32, initializer=tf.random.normal([]))
        with tf.variable_scope('trunk'):
            """Wide_layer is a basic LR model, including click order, transaction order,
            item statistical characteristics and query pair pairs. The deep model connects
            to two traditional attention models, and finally adds the output layer linearly.
            """
            if self.mode == 'train':
                sparse_features, itemid_emb, sequence_emb, sequence_token_mask, itemid_token_mask = args
                tfprint = tf.print(*[tf.shape(t) for t in args], 'shape', summarize=10000)
                with tf.control_dependencies([]):
                    sparse_features = tf.identity(sparse_features)
                # wide_layer
                # deep_layer
                # sequence_emb = tf.random.normal([self.batch_size, 20, self.cfg.embedding_dim])
                # itemid_emb = tf.random.normal([self.batch_size, self.embedding_size])
                # sequence_token_mask = tf.random.normal([self.batch_size, self.cfg.reduce_sequence_length])
                # itemid_token_mask = tf.random.normal([self.batch_size, 1])
            else:
                sparse_features, itemid_emb, sequence_emb, _, _ = args
                self.sequence = sequence_emb
                self.itemid = tf.squeeze(itemid_emb, axis=1)

                # self.sequence = tf.random.normal([None, self.cfg.reduce_sequence_length, self.cfg.embedding_dim], name='sequence')
                # self.itemid = tf.random.normal([None, self.cfg.embedding_dim], name='itemid')
                sequence_expand = tf.reduce_mean(self.sequence, 2)
                sequence_token_mask = tf.cast(sequence_expand, tf.bool)
                itemid_token_mask = tf.cast(tf.reduce_mean(tf.expand_dims(self.itemid, 1), 2), tf.bool)
            with tf.variable_scope("wide_layer") as scope:
                if self.mode == 'train':
                    self.add_sum = tf.reduce_sum(sparse_features, axis=1, name='wide_sum')
                    self.logits_wide = self.add_sum + self.b
                    self.ctr = tf.sigmoid(self.logits_wide)
                    print('logits_wide_shape', self.logits_wide.get_shape())

            with tf.variable_scope("deep_layer") as scope:
                if self.mode == 'train':
                    # sequence_emb = tf.nn.embedding_lookup(self.item_mat, tf.slice(self.sequence, [0, 0], [self.batch_size,
                    #                                                                                       self.cfg.reduce_sequence_length]))
                    # itemid_emb = tf.nn.embedding_lookup(self.item_mat, self.itemid)
                    # sequence_token_mask = tf.cast(
                    #     tf.slice(self.sequence, [0, 0], [self.batch_size, self.cfg.reduce_sequence_length]), tf.bool)
                    # itemid_token_mask = tf.cast(self.itemid, tf.bool)
                    self.logits_deep = self.bulid_attention_layers(sequence_emb, sequence_token_mask, itemid_emb,
                                                                   itemid_token_mask, 'attention')
                else:
                    # deep input
                    self.bs = tf.shape(self.itemid)[0]
                    print('bs_shape', self.bs)
                    self.logits_deep = self.bulid_attention_layers(self.sequence, sequence_token_mask,
                                                                   tf.expand_dims(self.itemid, 1),
                                                                   itemid_token_mask, 'attention')
                # self.logits_deep=tf.squeeze(logits_deep)
                print('logits_deep_shape', self.logits_deep.get_shape())

            with tf.variable_scope("output_layer"):
                if self.mode == 'train':
                    self.logits = self.logits_wide + self.logits_deep
                    tf.summary.histogram('logits_wide', self.logits_wide)
                    tf.summary.histogram('logits_deep', self.logits_deep)
                    tf.summary.histogram('logits', self.logits)
                else:
                    self.logits = self.logits_deep
                print('logits_shape', self.logits.get_shape())
            # self.y = tf.identity(self.logits_deep, name=self.out_node_names[0])
            self.predictions = tf.nn.sigmoid(self.logits)
        self.y = tf.identity(self.logits_deep, name=self.out_node_names[0])
        return self.predictions

    @staticmethod
    def _secondary_parse_fn(sparse_emb, deep_emb, dense_emb, mask, model_struct):
        wide = model_struct['wide']
        itemid = model_struct['trigger_id'][0]
        sequence = model_struct['seq'][0]
        tfprint = tf.print('cid', deep_emb['cid'], 'click_cross', deep_emb['click_cross'], 'order', deep_emb['order_cross'])
        with tf.control_dependencies([]):
            sparse_features = tf.concat([tf.squeeze(deep_emb[w], axis=1) for w in wide], axis=1)
        itemid_emb = deep_emb[itemid]
        sequence_emb = deep_emb[sequence]
        sequence_token_mask = mask[sequence]
        itemid_token_mask = tf.ones([tf.shape(mask[sequence])[0], 1])
        return sparse_features, itemid_emb, sequence_emb, sequence_token_mask, itemid_token_mask

    def get_eval_metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        # labels = tf.expand_dims(tf.cast(labels, tf.int32), axis=1)
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

    def bulid_optimize(self,learning_rate):

        # ---------- optimization ---------
        if self.optimizer_name.lower() == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer_name.lower() == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate)
        elif self.optimizer_name.lower() == 'ftrl':
            opt= tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                               initial_accumulator_value=1.0 / pow(self.batch_size, 2),
                                               l1_regularization_strength=self.l1 / self.batch_size,
                                               l2_regularization_strength=self.l2 / self.batch_size,
                                               use_locking=True)
        elif self.optimizer_name.lower() == 'combine':
            wide_opt=tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                               initial_accumulator_value=1.0 / pow(self.batch_size, 2),
                                               l1_regularization_strength=self.l1 / self.batch_size,
                                               l2_regularization_strength=self.l2 / self.batch_size,
                                               use_locking=True)

            #deep_opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
            deep_opt = tf.train.FtrlOptimizer(learning_rate=learning_rate,initial_accumulator_value=1.0 / pow(self.batch_size, 2),
                                              l1_regularization_strength=0.3/self.batch_size,
                                              l2_regularization_strength=5.0/self.batch_size,
                                              use_locking=True)

            opt=[wide_opt,deep_opt]
        elif self.optimizer_name.lower() == 'ftrl_adadelta':
            wide_opt=tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                               initial_accumulator_value=1.0 / pow(self.batch_size, 2),
                                               l1_regularization_strength=self.l1 / self.batch_size,
                                               l2_regularization_strength=self.l2 / self.batch_size,
                                               use_locking=True)

            deep_opt = tf.train.AdadeltaOptimizer(learning_rate= 1.0)

            opt=[wide_opt,deep_opt]

        elif self.optimizer_name.lower() == 'ftrl_adam':
            wide_opt = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                              initial_accumulator_value=1.0 / pow(self.batch_size, 2),
                                              l1_regularization_strength=self.l1 / self.batch_size,
                                              l2_regularization_strength=self.l2 / self.batch_size,
                                              use_locking=True)

            deep_opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)

            opt = [wide_opt, deep_opt]
        else:
            raise AttributeError('no optimizer named as \'%s\'' % self.optimizer_name)
        return opt

    def get_train_op_fn(self, loss, params):
        self.global_step = tf.compat.v1.train.get_global_step()
        self.loss = loss

        self.optimizer = self.bulid_optimize(self.lr)

        if self.optimizer_name.lower() in ['combine', 'ftrl_adadelta', 'ftrl_adam']:
            self.trainer = []
            self.wide_optimizer, self.deep_optimizer = self.optimizer[0], self.optimizer[1]
            wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "wide_layer")
            print('trainable wide var num: %d' % len(wide_vars))
            deep_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "deep_layer")
            print('trainable deep var num: %d' % len(deep_vars))
            self.trainer.append(self.wide_optimizer.minimize(self.loss, self.global_step, var_list=wide_vars))
            "Cross optimization"
            self.trainer.append(
                self.deep_optimizer.minimize(self.loss, self.global_step, var_list=deep_vars))
            """
            self.trainer.append(
                self.deep_optimizer.minimize(self.loss, self.global_step, var_list=deep_vars))
            """
            self.trainer = tf.group(*self.trainer)
        else:
            self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)
        return self.trainer

    def bulid_attention_layers(self, u_emb, all_token_mask, itemid_emb, temid_token_mask, scope):
        # partitioner = tf.fixed_size_partitioner(self.partition)
        with tf.variable_scope('attention'):#, partitioner=partitioner):
            if not self.mode == 'train':
                sequence_rep_single = single_attention(
                    u_emb, all_token_mask, scope + '_sequence', self.cfg.dropout, self.is_train, self.cfg.wd,
                    'elu', None, 's1_attention', is_multi_att=False, attention_dim=None
                )
                sequence_rep = tf.tile(sequence_rep_single, multiples=[self.bs, 1])
                tf.get_variable_scope().reuse_variables()
                itemid_rep = single_attention(
                    itemid_emb, temid_token_mask, scope + '_sequence', self.cfg.dropout, self.is_train, self.cfg.wd,
                    'elu', None, 's2_attention', is_multi_att=False, attention_dim=None
                )
            else:
                sequence_rep = single_attention(
                    u_emb, all_token_mask, scope + '_sequence', self.cfg.dropout, self.is_train, self.cfg.wd,
                    'elu', None, 's1_attention', is_multi_att=False, attention_dim=None
                )
                tf.get_variable_scope().reuse_variables()
                itemid_rep = single_attention(
                    itemid_emb, temid_token_mask, scope + '_sequence', self.cfg.dropout, self.is_train, self.cfg.wd,
                    'elu', None, 's2_attention', is_multi_att=False, attention_dim=None
                )
            out_rep = tf.concat([sequence_rep, itemid_rep, sequence_rep - itemid_rep, sequence_rep * itemid_rep],
                                1)  # [bs,128*4]
        with tf.variable_scope('output_layer_last'):#, partitioner=partitioner):
            attention_layer = dice(
                linear([out_rep], self.cfg.hidden_num, True, 0., scope=scope + '_output_attention_layer', squeeze=False,
                       wd=self.cfg.wd, input_keep_prob=self.cfg.dropout, is_train=self.is_train),
                name='attention_pre_out')

            # logits_deep_dropout = tf.nn.dropout(attention_layer,self.cfg.last_dropout)
            # logits_deep = tf.layers.dense(inputs=logits_deep_dropout, units=1, activation=None, use_bias=True)[:,0]
            logits_deep = linear([attention_layer], 1, True, 0., scope=scope + '_logits', squeeze=False,
                                 wd=self.cfg.wd, input_keep_prob=self.cfg.last_dropout, is_train=self.is_train)[:, 0]
        self.logits = logits_deep
        self.predictions = tf.nn.sigmoid(logits_deep)
        return self.predictions

    def get_loss(self, labels, predictions):
        """Using sigmoid_cross_entropy_with_logits, the shape constraints of label and Logits are consistent and one-dimensional array."""

        # trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'deep_layer')
        # print('trainable var num: %d' % len(trainable_vars))
        labels = tf.cast(labels, tf.float32, name='true_label')
        labels = tf.squeeze(labels, axis=1)
        labels = tf.slice(labels, [0], tf.shape(self.logits))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
        loss = tf.reduce_mean(losses, name='loss')
        tf.summary.scalar('loss', loss)

        # weights = tf.slice(self.reweight, [0], tf.shape(self.logits))
        self.loss_sum = tf.reduce_sum(losses, name='loss_sum')
        # self.weight_sum = tf.reduce_sum(weights, name='weight_sum')
        tf.summary.scalar('loss_sum', self.loss_sum)
        # tf.summary.scalar('weight_sum', self.weight_sum)
        return loss


def single_attention(rep_tensor, rep_mask, scope=None,
                    keep_prob=1., is_train=None, wd=0., activation='elu',
                    tensor_dict=None, name='',is_multi_att=False, attention_dim=None):

    """single attention: contain two traditional attention struct.

    Args:
      rep_tensor: list tensor,shape is [batch_size,list_size,embedding_dim].
      rep_mask: whether the marker Tensor is 0,bool value, shape is [batch_size,list_size]
    Returns:
      Attention representation of tensor,shape is [batch_size,embedding_dim]
    Raises:
      TypeError: If the input dimension is incorrect.
    """

    with tf.variable_scope(scope):
        attention_rep_first_layer = traditional_attention(
            rep_tensor, rep_mask, 'traditional_attention',
            keep_prob, is_train, wd, activation,
            tensor_dict=tensor_dict, name=name + '_attn')

        #attention_rep_final = traditional_attention(
        #    attention_rep_first_layer, rep_mask, 'traditional_attention',
        #    keep_prob, is_train, wd, activation,
        #    tensor_dict=tensor_dict, name=name + '_attn')

        return attention_rep_first_layer

def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0,
               input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "linear"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()
def dice(_x, axis=-1, epsilon=0.0000001, name=''):

    alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape())
    reduction_axes = list(range(len(input_shape)))

    del reduction_axes[axis]  # [0]

    broadcast_shape = [1] * len(input_shape)  # [1,1]
    broadcast_shape[axis] = input_shape[axis]  # [1 * hidden_unit_size]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)  # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)  # [1 * hidden_unit_size]
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x

def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None,output_dim=None):

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    if output_dim is None:
        ivec = rep_tensor.get_shape()[2]
    else:
        ivec = output_dim
    with tf.variable_scope(scope):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)
        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        #if tensor_dict is not None and name is not None:
        #    tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res

def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train, scope='bn')
        return activation_func(linear_map)


def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out

def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (isinstance(args, (tuple, list)) and not args):
            raise ValueError("`_args` must be specified")
        if not isinstance(args, (tuple, list)):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits,-1)
        return out

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`_args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.layers.dropout(arg, rate=input_keep_prob, training=is_train) for arg in flat_args]
        # flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
        #              for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def _linear(xs,output_size,bias,bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size],dtype=tf.float32,)
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)

