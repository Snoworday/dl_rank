

GRAPH_PB_PATH = '/Users/snoworday/git/algo-deeplearning/result/graph/wide_and_deep_traditional_attention_v2/pbtxt'

import tensorflow as tf






with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], GRAPH_PB_PATH)
    trainable_coll = sess.graph.get_collection(tf.compat.v1.GraphKeys.VARIABLES)
    # out_tensor = sess.graph.get_tensor_by_name("trunk/fm/batch_normalization/moving_mean/read:0")
    # print(sess.run(out_tensor))
    # es = [v.name for v in tf.compat.v1.trainable_variables()]
    for v in trainable_coll:
        print(sess.run(v.value()))



with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], GRAPH_PB_PATH)
    trainable_coll = sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    es = [v.name for v in tf.compat.v1.trainable_variables()]
    for v in trainable_coll:
        tf.compat.v1.get_default_graph().as_graph_element(es[0])
        q=1
        print(sess.run(v.name))

##

from tensorflow.python import pywrap_tensorflow
checkpoint_path = '/Users/snoworday/data/deepfm/deepfm/model.ckpt-8100'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names
