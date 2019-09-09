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

python3 -m dl_rank.solo --mode train --tw 3 --wi 0 --useps  --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_*_172.31.27.48:2222_]*_ps_:[_172.31.21.20:2222_]}*_task_:{_type_:_chief_*_index_:0}} --logpath /home/hadoop

python3 -m dl_rank.solo --mode train --tw 3 --wi 1 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_*_172.31.27.48:2222_]*_ps_:[_172.31.21.20:2222_]}*_task_:{_type_:_worker_*_index_:0}} --logpath /home/hadoop

python3 -m dl_rank.solo --mode train --tw 3 --wi 2 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_*_172.31.27.48:2222_]*_ps_:[_172.31.21.20:2222_]}*_task_:{_type_:_worker_*_index_:1}} --logpath /home/hadoop

python3 -m dl_rank.solo --mode train --tw 3 --wi 0 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_*_172.31.27.48:2222_]*_ps_:[_172.31.21.20:2222_]}*_task_:{_type_:_ps_*_index_:0}} --logpath /home/hadoop




# stable
python3 /home/hadoop/_dl_rank/solo.py --mode train --tw 2 --wi 0 --useps --conf I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_chief_*_index_:0}} --logpath /home/hadoop
python3 /home/hadoop/_dl_rank/solo.py --mode train --tw 2 --wi 1 --useps --conf I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_worker_*_index_:0}} --logpath /home/hadoop
python3 /home/hadoop/_dl_rank/solo.py --mode train --tw 2 --wi 0 --useps --conf I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_ps_*_index_:0}} --logpath /home/hadoop
python3 /home/hadoop/_dl_rank/solo.py --mode train --tw 1 --wi 0 --useps --conf I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_evaluator_*_index_:0}} --logpath /home/hadoop

# module
python3 -m dl_rank.solo --mode train --tw 2 --wi 0 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_chief_*_index_:0}} --logpath /home/hadoop
python3 -m dl_rank.solo --mode train --tw 2 --wi 1 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_worker_*_index_:0}} --logpath /home/hadoop
python3 -m dl_rank.solo --mode train --tw 2 --wi 0 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_ps_*_index_:0}} --logpath /home/hadoop
python3 -m dl_rank.solo --mode train --tw 1 --wi 0 --useps --conf /home/hadoop/_dl_rank/conf/I2Iconf_uv --tfconfig {_cluster_:{_chief_:[_172.31.25.189:2222_]*_worker_:[_172.31.21.154:2222_]*_ps_:[_172.31.27.48:2222_]}*_task_:{_type_:_evaluator_*_index_:0}} --logpath /home/hadoop


${SPARK_HOME}/bin/spark-submit --master yarn --py-files /home/hadoop/_dl_rank/conf/I2Iconf_uv/__pycache__,/home/hadoop/_dl_rank/conf/I2Iconf_uv/sparator.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/train.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/schema.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/.DS_Store,/home/hadoop/_dl_rank/conf/I2Iconf_uv/model.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/feature.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/item_profile_parse.py,/home/hadoop/_dl_rank/conf/I2Iconf_uv/vocabulary.yaml,/home/hadoop/_dl_rank/conf/I2Iconf_uv/parser.py  --num-executors 4 --executor-memory 15G --driver-memory 15G --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" --conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" --conf spark.executorEnv.PYSPARK_DRIVER_PYTHON="/usr/bin/python3"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}"  --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" /usr/local/lib/python3.6/site-packages/dl_rank-0.1.5-py3.6.egg/dl_rank/solo.py --date 2019-08-14 --mode infer --useSpark --logpath /home/hadoop --conf _dl_rank/conf/I2Iconf_uv

