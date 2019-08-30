import tensorflow as tf
from tensorflow.python.tools import freeze_graph
tmp2 = 3

data_dir = "/Users/snoworday/git/algo-deeplearning/result/graph/wide_and_deep_traditional_attention/"
GRAPH_PB_PATH = data_dir # + 'resnet_v2_fp32_savedmodel_NCHW/1538687196/'




with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], graph_pb_path)
    # trainable_coll = sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    # # es = [v.name for v in tf.compat.v1.trainable_variables()]
    # for v in trainable_coll:
    #     print(sess.run(v.name))#, sess.run(v.value()))
    #     print(sess.run(v.value()))

    freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False,
                              input_checkpoint=ckpt_filepath, output_node_names='cnn/output',
                              restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph=pb_filepath, clear_devices=True, initializer_nodes='')

for n in tf.compat.v1.get_default_graph()._nodes_by_name:
    print(tf.compat.v1.get_default_graph()._nodes_by_name[n].outputs)



from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util
import copy
import tensorflow as tf
slim = tf.contrib.slim

Output_node_names = 'output'
Output_graph_name = '/Users/snoworday/Documents'
logdir = ''
checkpoint_file = tf.train.latest_checkpoint(logdir)
orgname = ['trunk/Squeeze']
newname = ['placeholder/itemid']
replace = dict(zip(orgname, newname))

org_graph = tf.Graph()
new_graph = tf.Graph()

with org_graph.as_default() as graph:
    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], GRAPH_PB_PATH)
    org_graph_def = tf.get_default_graph().as_graph_def()

new_graph_def = graph_pb2.GraphDef()
for org_node in org_graph_def.node:
    if org_node.name in orgname:
        new_op = tf.placeholder(dtype=org_node.dtype, shape=org_node.shape, name=replace[org_node.name])
        new_graph_def.node.extend([new_op])
        continue
    for node_input in org_node.input:
        if node_input.name in orgname:
            org_node.input.remove(node_input.name)
            org_node.input.extend([replace[node_input.name]])
    new_graph_def.node.extend([copy.deepcopy(org_node)])

with new_graph.as_default() as graph:
    tf.import_graph_def(new_graph_def)
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    input_graph_def = new_graph_def
    output_node_names = Output_node_names
    output_graph_name = Output_graph_name
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())






