#!/usr/bin/env bash
sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
spark-submit \
--master yarn \
--num-executors 80 \
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
s3://jiayun.spark.data/wangqi/datapipeline/gen_wide_deep_dataset.py
aws s3 cp /tmp/profile* s3://jiayun.spark.data/wangqi/wide_deep/Feature/profile/
