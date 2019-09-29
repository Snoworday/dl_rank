#!/usr/bin/env bash
sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
aws s3 cp s3://jiayun.spark.data/wangqi/datapipeline/infer_ctr_uv.py ./
aws s3 rm --recursive s3://jiayun.spark.data/wangqi/I2IRank/resultJson
sudo python3
python3 ./infer_ctr_uv.py
