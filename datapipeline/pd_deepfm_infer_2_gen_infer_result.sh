#!/usr/bin/env bash
sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh

aws s3 cp s3://jiayun.spark.data/wangqi/datapipeline/infer_ctr_uv.py ./
#mkdir I2Iconf_uv
aws s3 cp s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/I2Iconf_uv ./I2Iconf_uv --recursive
echo $PWD
#python3 ./infer_ctr_uv.py


ls -al

aws emr list-clusters

