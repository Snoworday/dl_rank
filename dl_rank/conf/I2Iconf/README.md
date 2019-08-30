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
generate_train_data_v2.py \
--date 2019-08-27 \
> tmp.log 2>&1 &


