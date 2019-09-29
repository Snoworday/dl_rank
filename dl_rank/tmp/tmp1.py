import sys

path = '''
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
s3://jiayun.spark.data/wangqi/datapipeline/gen_ctr_json_from_infer.py
'''

if __name__ == '__main__':
    path1 = path
    file1 = path#open(path1, "r").read()
    lis1 = file1.split("spark-submit")
    lis2 = []
    for line in lis1:
        lis = line.replace("\t", " ").replace("\\", " ").replace("\n", " ").split(" ")
        lis = [v for v in lis if (not "".__eq__(v))]
        lis2.append("command-runner.jar,spark-submit," + ",".join(lis))

    for v in lis2:
        print(v)