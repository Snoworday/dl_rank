sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
export SPARK_HOME=/usr/lib/spark
export HADOOP_HOME=/usr


# Item Profile Train Data Generate
spark-submit \
--master yarn \
--num-executors 30 \
--executor-memory 15G \
--driver-memory 15G \
--executor-cores 4 \
--py-files sparator.py \
--conf spark.executorEnv.SPARK_HOME="/usr/lib/spark" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.storage.memoryFraction=0.7 \
--conf spark.shuffle.memoryFraction=0.2 \
--conf spark.default.parallelism=100 \
--conf spark.core.connection.ack.wait.timeout=5000 \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="${CLASSPATH}" \
item_profile_parse.py \
--mode infer \
--date 2019-08-13

date=2019-08-13
conf=I2Iconf_uv
executor=40
# infer
nohup ${SPARK_HOME}/bin/spark-submit \
--master yarn \
--py-files I2Iconf_uv.zip,conf.zip,model.zip,utils.zip,sparkDlManager.py,I2Iconf_uv/feature.yaml,I2Iconf_uv/schema.yaml,I2Iconf_uv/model.yaml,I2Iconf_uv/vocabulary.yaml,I2Iconf_uv/train.yaml  \
--files I2Iconf_uv/feature.yaml,I2Iconf_uv/schema.yaml,I2Iconf_uv/model.yaml,I2Iconf_uv/vocabulary.yaml,I2Iconf_uv/train.yaml \
--num-executors 40 \
--executor-memory 15G \
--driver-memory 15G \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON="/usr/bin/python3" \
--conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" \
main.py \
--date 2019-08-13 \
--mode infer \
--useSpark True \
--conf I2Iconf_uv  > tmp.log 2>&1 &





# Result Json Data Generate
nohup spark-submit \
--master yarn \
--num-executors 40 \
--executor-memory 15G \
--driver-memory 15G \
--executor-cores 4 \
--conf spark.executorEnv.SPARK_HOME="/usr/lib/spark" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.storage.memoryFraction=0.7 \
--conf spark.shuffle.memoryFraction=0.2 \
--conf spark.default.parallelism=100 \
--conf spark.core.connection.ack.wait.timeout=5000 \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="${CLASSPATH}" \
csvResult2json.py \
--date 2019-08-13 \
--datapath s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/result \
--output s3://jiayun.spark.data/wangqi/I2IRank/resultJson > tmp.log 2>&1 &


# gen wid_deep data
nohup spark-submit \
--master yarn \
--num-executors 40 \
--executor-memory 15G \
--driver-memory 15G \
--executor-cores 4 \
--conf spark.driver.maxResultSize="2g" \
--conf spark.executorEnv.SPARK_HOME="/usr/lib/spark" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.storage.memoryFraction=0.7 \
--conf spark.shuffle.memoryFraction=0.2 \
--conf spark.default.parallelism=100 \
--conf spark.core.connection.ack.wait.timeout=5000 \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="${CLASSPATH}" \
generate_train_data_v3.py \
> tmp.log 2>&1 &
########
sudo rm -r /usr/local/lib/python3.6/site-packages/dl_rank*
python3 setup.py build
sudo python3 setup.py install
#######
ssh -i ~/wangqi.pem hadoop@ec2-34-216-139-139.us-west-2.compute.amazonaws.com
ssh -i ~/wangqi.pem hadoop@172.31.21.29

ssh -i ~/wangqi.pem hadoop@ec2-34-216-139-139.us-west-2.compute.amazonaws.com
ssh -i ~/wangqi.pem hadoop@172.31.20.62

ssh -i ~/wangqi.pem hadoop@ec2-34-216-139-139.us-west-2.compute.amazonaws.com
ssh -i ~/wangqi.pem hadoop@172.31.18.185
#######
scp -i ~/wangqi.pem /usr/local/lib/python3.6/site-packages/dl_rank-0.1.10-py3.6.egg/dl_rank/solo.py hadoop@172.31.21.29:/home/hadoop
scp -i ~/wangqi.pem /usr/local/lib/python3.6/site-packages/dl_rank-0.1.10-py3.6.egg/dl_rank/solo.py hadoop@172.31.20.62:/home/hadoop
scp -i ~/wangqi.pem /usr/local/lib/python3.6/site-packages/dl_rank-0.1.10-py3.6.egg/dl_rank/solo.py hadoop@172.31.18.185:/home/hadoop

sudo mv /home/hadoop/solo.py /usr/local/lib/python3.6/site-packages/dl_rank-0.1.10-py3.6.egg/dl_rank/solo.py

######
scp -i ~/wangqi.pem -r /home/hadoop/dl_rank hadoop@172.31.21.29:/home/hadoop
scp -i ~/wangqi.pem -r /home/hadoop/setup.py hadoop@172.31.21.29:/home/hadoop

scp -i ~/wangqi.pem -r /home/hadoop/dl_rank hadoop@172.31.20.62:/home/hadoop
scp -i ~/wangqi.pem -r /home/hadoop/setup.py hadoop@172.31.20.62:/home/hadoop

scp -i ~/wangqi.pem -r /home/hadoop/dl_rank hadoop@172.31.18.185:/home/hadoop
scp -i ~/wangqi.pem -r /home/hadoop/setup.py hadoop@172.31.18.185:/home/hadoop


${SPARK_HOME}/bin/spark-submit --master yarn \
--py-files /home/hadoop/dl_rank_/conf/I2Iconf_test_emr/sparator.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/train.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/schema.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/model.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/feature.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/item_profile_parse.py,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/vocabulary.yaml,/home/hadoop/dl_rank_/conf/I2Iconf_test_emr/parser.py \
--num-executors 3 --executor-memory 15G --driver-memory 15G \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON="/usr/bin/python3"  \
--conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):$(hadoop classpath --glob)"  \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${JAVA_HOME}/lib/amd64/server" \
--conf spark.yarn.appMasterEnv.S3_REQUEST_TIMEOUT=60000 \
/usr/local/lib/python3.6/site-packages/dl_rank-0.1.10-py3.6.egg/dl_rank/solo.py \
--date 2019-07-28 --mode infer --useSpark --logpath /home/hadoop --ps 1 --num_executor 3 --use_TFoS \
--conf /home/hadoop/dl_rank_/conf/I2Iconf_test_emr



tf.gfile.Exists('s3://jiayun.spark.data/wangqi/demo/I2IRank/I2ICTRUV/logData/deepfm_v2_mba/model.ckpt-8977.index')
