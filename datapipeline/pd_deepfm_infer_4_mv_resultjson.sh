#!/usr/bin/env bash

log=$(date -d "now" +%Y-%m-%d)
aws s3 rm s3://jiayun.spark.data/wangqi/I2IRank/resultJson
aws s3 cp s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/resultJson/${log} s3://jiayun.spark.data/wangqi/I2IRank/resultJson --recursive
