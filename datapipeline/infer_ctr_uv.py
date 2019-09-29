import dl_rank
import datetime

today = '2019-09-26'    # str(datetime.datetime.today()).split(' ')[0]
dl_rank.set_env(logpath='/mnt/taskRunner', use_TFoS=True)
dl_rank.set_env(conf='s3://jiayun.spark.data/wangqi/I2IRank/I2ICTRUV/I2Iconf_uv', date=today)
dl_rank.infer(executer_num=20)
dl_rank.wait(30)
print('finish')
