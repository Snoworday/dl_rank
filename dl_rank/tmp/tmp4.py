from tensorflow.python.framework import graph_util
import os
import tensorflow as tf

from utils.modelUtils import tensorflow_get_weights, tensorflow_set_weights

slim = tf.contrib.slim

Output_node_names = ['trunk/deep_layer/Sigmoid']
save_predict_dir = '/Users/snoworday/git/algo-deeplearning/result/graph/tmp'
if not os.path.exists(save_predict_dir):
    os.mkdir(save_predict_dir)

Output_variable_name = 'data.ckpt'
Output_graph_name = 'model.pb'

graph_pb_path = '/Users/snoworday/git/algo-deeplearning/result/graph/wide_and_deep_traditional_attention_v2/pbtxt'
logdir = '/Users/snoworday/git/algo-deeplearning/result/summary/wide_and_deep_traditional_attention_v2'
checkpoint_path = tf.train.latest_checkpoint(logdir)
orgname = ['trunk/Squeeze', 'embedding_lookup_2/Identity', 'ones', 'NotEqual_9']
newname = ['placeholder/itemid', 'placeholder/sequence', 'placeholder/itemid_mask', 'placeholder/sequence_mask']
replace = dict(zip(orgname, newname))

def export_online(save_predict_dir, output_node_names, replace_map, graph_pb_path='', checkpoint_path='',
                  output_variable_name='data.ckpt', output_graph_name='model.pb'):
    assert graph_pb_path != '' or checkpoint_path != '', 'Give me ur graph and vars, OAO'
    org_graph = tf.Graph()
    new_graph = tf.Graph()
    with tf.Session(graph=org_graph) as sess:
        if graph_pb_path != '':
            org_meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], graph_pb_path)
            weights = tensorflow_get_weights(sess)
        else:
            org_meta_graph_def = tf.train.latest_checkpoint(checkpoint_path) + '.meta'
            _ = tf.train.import_meta_graph(org_meta_graph_def)
            a = 1


    with tf.Session(graph=new_graph) as sess:
        input_map = dict()
        for org_name in replace_map:
            org_tensor = org_graph.get_tensor_by_name(org_name+':0')
            input_map[org_name+':0'] = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape, name=replace[org_name])
        new_meta_graph_def = tf.train.import_meta_graph(org_meta_graph_def, input_map=input_map, return_elements=output_node_names)
        if checkpoint_path!='':
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
        else:
            tensorflow_set_weights(sess, weights)
            saver = tf.train.Saver()
        saver.save(sess, save_path=os.path.join(save_predict_dir, output_variable_name))

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names
        )
        output_graph_path = os.path.join(save_predict_dir, output_graph_name)
        with tf.gfile.GFile(output_graph_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

export_online(save_predict_dir, Output_node_names, replace, graph_pb_path=graph_pb_path, checkpoint_path='')


#     org_graph_def = tf.get_default_graph().as_graph_def()
    # coll = sess.graph.get_collection(tf.compat.v1.GraphKeys.VARIABLES)
    # for v in coll:
    #     print(sess.run(v.value()))


# with tf.Session(graph=new_graph) as sess:
#     input_map = dict()
#     for org_name in orgname:
#         org_tensor = org_graph.get_tensor_by_name(org_name+':0')
#         input_map[org_name+':0'] = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape, name=replace[org_name])
#
#     new_saver = tf.train.import_meta_graph('/Users/snoworday/git/algo-deeplearning/result/summary/wide_and_deep_traditional_attention_v2/model.ckpt-235285.meta',
#                                            input_map=input_map)
#     new_saver.restore(sess, checkpoint_path)
#     a = 1
#


# with new_graph.as_default():
#     org_graph_node_name_list = [node.name for node in org_graph.as_graph_def().node]
#     for org_node in org_graph_def.node:
#         if org_node.name in orgname:
#             org_tensor = org_graph.get_tensor_by_name(org_node.name+':0')
#             new_tensor = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape, name=replace[org_node.name])
#             continue
#         for idx, node_input in enumerate(org_node.input):
#             if node_input in orgname:
#                 org_graph.as_graph_def().node[org_graph_node_name_list.index(org_node.name)].input[idx] = replace[node_input]
#                 # org_node.input[idx] = replace[node_input]
#     org_graph.as_graph_def().node.extend(new_graph.as_graph_def().node)

from tensorflow.python.framework import ops, meta_graph

with tf.Session(graph=new_graph) as sess:
    input_map = dict()
    for org_name in orgname:
        org_tensor = org_graph.get_tensor_by_name(org_name+':0')
        input_map[org_name+':0'] = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape, name=replace[org_name])
    new_meta_graph_def = tf.train.import_meta_graph(org_meta_graph_def, input_map=input_map, return_elements=Output_node_names)

    tensorflow_set_weights(sess, weights)
    # saver = tf.train.Saver(new_meta_graph_def.saver_def)
    # saver.restore(sess, )


    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        Output_node_names
    )
    output_graph_name = os.path.join(save_predict_dir, 'model.pb')
    with tf.gfile.GFile(output_graph_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())



with tf.Session(graph=org_graph) as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], graph_pb_path)
    for org_name in orgname:
        org_tensor = org_graph.get_tensor_by_name(org_name+':0')
        new_tensor = tf.placeholder(dtype=org_tensor.dtype, shape=org_tensor.shape, name=replace[org_name])
    for org_node in org_graph.as_graph_def().node:
        for idx, node_input in enumerate(org_node.input):
            if node_input in orgname:
                org_graph._nodes_by_name[org_node.name].inputs._inputs[idx] = org_graph.get_tensor_by_name(replace[node_input]+':0')
                # org_node.input[idx] = replace[node_input]

    output_graph_name = os.path.join(save_predict_dir, 'model.pb')

    try:
        variables_to_restore = org_graph.get_collection(tf.GraphKeys.VARIABLES)
        saver = tf.train.Saver(variables_to_restore)
    except:
        saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        org_graph.as_graph_def(),
        Output_node_names
    )
    with tf.gfile.GFile(output_graph_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

with tf.Graph().as_default() as new_graph:
    tf.import_graph_def(output_graph_def)
    variables_to_restore = new_graph.get_collection(tf.GraphKeys.VARIABLES)
    a=1


with tf.Graph().as_default() as new_graph:
    tf.import_graph_def(output_graph_def)
    try:
        variables_to_restore = new_graph.get_collection(tf.GraphKeys.VARIABLES)
        saver = tf.train.Saver(variables_to_restore)
    except:
        saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        saver.save(os.path.join(save_predict_dir, 'data.ckpt'))

print('finish!')

