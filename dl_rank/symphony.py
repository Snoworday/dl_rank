from __future__ import absolute_import
import paramiko
import os
import json
import inspect
import subprocess
import argparse
from tensorflow import gfile

try:
    from dl_rank.utils.modelUtils import setEnv
except:
    from utils.modelUtils import setEnv
from multiprocessing import Pool

# names about EMR
UserName = 'hadoop'
port = 22
tf_port = 2222
emr_home_path = '/home/hadoop'

_project_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_exec_file = 'solo.py'


class SFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are
            created under target.
        '''
        if os.path.isfile(source):
            try:
                self.put(source, '%s' % target)
            except:
                pass
        else:
            for item in os.listdir(source):
                if os.path.isfile(os.path.join(source, item)):
                    try:
                        self.put(os.path.join(source, item), '%s/%s' % (target, item))
                    except:
                        pass
                else:
                    self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                    self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=True):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(SFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise

def getClusterId(emrName):
    cmd = r'''
    aws emr list-clusters  --active | jq '.Clusters' | jq '.[]' -c | grep  {} | jq '.Id' | tr -d '"'
    '''.format(emrName)
    cmd_out = os.popen(cmd).read()
    clusterId = cmd_out.split('\n')[0]
    return clusterId

def getHostnameList_Chief_Hostnum(emrName):
    cmd = r'''
    aws emr list-instances --cluster-id {} | jq '.Instances' | jq '.[]' -c | jq '.PrivateIpAddress' | tr -d '"'
    '''.format(getClusterId(emrName))
    cmd_out = os.popen(cmd).read()
    hostnameList = cmd_out.split('\n')[:-1]

    cmd = r'''
    aws emr list-instances --cluster-id {} | grep 'PrivateIpAddress' | wc -l
    '''.format(getClusterId(emrName))
    cmd_out = os.popen(cmd).read()
    hostnum = cmd_out.split('\n')[0]

    cmd = r'''
    ifconfig | grep 'inet addr:172' | awk '{print $2}' | sed 's/addr://g'
    '''
    cmd_out = os.popen(cmd).read()
    chief = cmd_out.split('\n')[0]
    return hostnameList, chief, hostnum

def generateTFConfig(worker_ps_evaluator, chief, ps_num, ev_num):
    def addPort(nodelist, port=tf_port):
        out = [n+':'+str(port) for n in nodelist]
        return out
    assert ps_num + ev_num <= len(worker_ps_evaluator), 'no sufficient node remaining'
    if ev_num == 0:
        worker, ps = worker_ps_evaluator[:-(ps_num + ev_num)], worker_ps_evaluator[-(
                    ps_num + ev_num):]
        evaluator = []
    else:
        worker, ps, evaluator = worker_ps_evaluator[:-(ps_num+ev_num)], worker_ps_evaluator[-(ps_num+ev_num):-ev_num], worker_ps_evaluator[-ev_num:]

    TF_CONFIG_dict = dict()
    cluster = {'chief': addPort([chief]), 'worker': addPort(worker)}
    if ps_num:
        cluster.update({'ps': addPort(ps)})
    for type, nodes in cluster.items():
        idx = 0
        for node in nodes:
            task = {'type': type, 'index': idx}
            TF_CONFIG_dict[node.split(':')[0]] = {'cluster': cluster, 'task': task}
            idx += 1
    idx = 0
    for node in evaluator:
        task = {'type': 'evaluator', 'index': idx}
        TF_CONFIG_dict[node] = {'cluster': cluster, 'task': task}
        idx += 1
    return TF_CONFIG_dict

def transportFile(ssh, file_path_list, target_path):
    transport = ssh.get_transport()
    sftp = SFTPClient.from_transport(transport)
    sftp.mkdir(target_path)
    for file_path in file_path_list:
        target_path = os.path.join(target_path, file_path.rsplit('/', 1)[1])
        sftp.put_dir(file_path, target_path)

def distributeMulti(mode, *args, **kwargs):
    func = distributeTrain if mode == 'train' else distributeInfer
    p = Pool()
    result = p.apply(func, args, kwargs)
    p.close()
    p.join()
    pid = result
    return pid

def distributeTrain(confDir, date, retrain, ps_num, ev_num, emrName, keyFile, logDir=emr_home_path, dirs2issue=None):
    if confDir[-1] == '/':
        confDir = confDir[:-1]
    if retrain:
        clear_summary(confDir, logDir)
    if dirs2issue is None:
        dirs2issue = []
    hostnameList, chief, hostnum = getHostnameList_Chief_Hostnum(emrName)
    hostnameList.pop(hostnameList.index(chief))
    TF_CONFIG_dict = generateTFConfig(hostnameList, chief, ps_num, ev_num)
    total_worker = int(hostnum) - ev_num - ps_num
    worker_wi, ps_wi = 0, 0
    date = '--date {}'.format(date) if date!='' else ''
    confName = confDir.rsplit('/', 1)[1] if ('/' in confDir) and not confDir.startswith('s3') else confDir
    if confName != confDir:
        dirs2issue.append(confDir)

    def run_remote(wi, TF_CONFIG, UserName, hostname, is_eval):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=UserName, key_filename=keyFile)
        if not dirs2issue:
            transportFile(ssh, dirs2issue, logDir)
        if gfile.Exists(os.path.join(os.getcwd(), confDir)):
            transportFile(ssh, os.path.join(os.getcwd(), confDir), logDir)
        if __name__=='__main__':
            trainpath = os.path.join(logDir, dirs2issue[0].split('/')[-1], _exec_file)
        else:
            trainpath = '-m dl_rank.solo'
        cmd = r'''
        python3 {trainpath} --mode train --tw {tw} --wi {wi} --ps {ps} {date} --conf {conf} --tfconfig {tfconfig} --logpath {logpath}
        '''.format(trainpath=trainpath, tw=total_worker if not is_eval else 1, wi=wi, ps=ps_num,
                   date=date, conf=confName, tfconfig=TF_CONFIG, logpath=logDir)
        nohup_cmd = '/home/hadoop/nohup' + cmd + ' > {} 2>&1 &'.format(os.path.join(logDir, 'dl_rank_run.log'))
        cmd = cmd + ' > {}'.format(os.path.join(logDir, 'dl_rank_run.log'))
        # subprocess.Popen(['ssh', ' -i {} '.format(keyFile), UserName + '@' + hostname, nohup_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _ = ssh.exec_command(cmd)
        ssh.close()

    def run_local(wi, TF_CONFIG):
        if __name__ == '__main__':
            trainpath = os.path.join(dirs2issue[0], _exec_file)
        else:
            trainpath = '-m dl_rank.solo'
        cmd = '''
        python3 {trainpath} --mode train --tw {tw} --wi {wi} --ps {ps} {date} --conf {conf} --tfconfig {tfconfig} --logpath {logpath}
        '''.format(trainpath=trainpath, tw=total_worker, wi=wi, ps=ps_num, date=date,
                   conf=confDir, tfconfig=TF_CONFIG, logpath=logDir)
        # nohup_cmd = '/home/hadoop/nohup' + cmd + ' > {} 2>&1 &'.format(os.path.join(logDir, 'dl_rank_run.log'))
        cmd = cmd + ' > {}'.format(os.path.join(logDir, 'dl_rank_run.log'))
        process = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        return process

    for hostname, TF_CONFIG in TF_CONFIG_dict.items():
        node_type = TF_CONFIG['task']['type']
        if node_type in ['chief', 'worker']:
            wi = worker_wi
            worker_wi += 1
        elif node_type == 'ps':
            wi = ps_wi
            ps_wi += 1
        else:
            wi = 0
        TF_CONFIG = json.dumps(TF_CONFIG).replace(' ', '').replace('"', '_').replace(',', '*')
        if node_type == 'chief':
            chief = run_local(wi, TF_CONFIG)
        else:
            run_remote(wi, TF_CONFIG, UserName, hostname, is_eval=node_type == 'evaluator')
    return chief.pid

def stopCluster(emrName, keyFile):
    hostnameList, chief, _ = getHostnameList_Chief_Hostnum(emrName)
    for node in hostnameList:
        if node == chief:
            os.system("ps -ef | grep python | grep solo | awk '{print \"kill -9 \" $2}' | bash -v")
        else:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=node, port=port, username=UserName, key_filename=keyFile)
            _ = ssh.exec_command("ps -ef | grep python | grep solo | awk '{print \"kill -9 \" $2}' | bash -v")
            ssh.close()

def _zipDir(dirList, project_path):
    from zipfile import ZipFile, ZIP_DEFLATED
    py_file = []
    for dir in dirList:
        dir_path = os.path.join(project_path, '{}'.format(dir))
        if os.path.exists(dir_path + '.zip'):
            os.remove(dir_path + '.zip')
        zipfile = ZipFile(dir_path+'.zip', 'w', ZIP_DEFLATED)
        for file in os.listdir(dir_path):
            zipfile.write(os.path.join(dir_path, file))
        zipfile.close()
        py_file.append(dir_path + '.zip')
    return py_file

def _find_conf(conf, save_path):
    if conf.startswith('s3'):
        if not gfile.Exists(conf):
            conf = conf.strip('/')
            conf_name = conf.rsplit('/', 1)[1]
            save_path = os.path.join(save_path, conf_name) + '/'
            gfile.MakeDirs(save_path)
            check = os.system('aws s3 cp {} {} --recursive'.format(conf+'/', save_path))
            conf = save_path
            if check:
                assert False, 'cant find conf in: {}'.format(conf)
        return conf
    conf_path = os.path.abspath(conf)
    if not gfile.Exists(conf_path):
        conf_path = os.path.join(_project_path, 'conf', conf)
    return conf_path


def _addConfig(conf_path):
    if conf_path.startswith('s3'):
        return []
    py_files = []
    for file in os.listdir(conf_path):
        py_files.append(os.path.join(conf_path, file))
    assert py_files != [], 'Cant find conf dir'
    return py_files

def clear_summary(conf, bake_path):
    import yaml
    conf_path = _find_conf(conf, bake_path)
    with open(conf_path + '/mission.yaml', 'r') as f:
        train_conf = yaml.load(f)
        model_dir = train_conf['train']['model_dir']
    if gfile.Exists(model_dir):
        gfile.DeleteRecursively(model_dir)

def run_tensorboard(conf='', model_dir='', logpath=emr_home_path, port=6006):
    assert conf!='' or model_dir!='', 'Please define summary path'
    if model_dir != '':
        pass
    else:
        import yaml
        conf_path = _find_conf(conf, logpath)
        with gfile.GFile(conf_path + '/mission.yaml', 'r') as f:
            train_conf = yaml.load(f)
            model_dir = train_conf['train']['model_dir']

    nohup_cmd = '/usr/bin/nohup python3 -m tensorboard.main --logdir={model_dir} --port={port} > {logpath} 2>&1 &'.format(
        model_dir=model_dir, port=port, logpath=os.path.join(logpath, 'dl_rank_tensorboard.log'))
    process = subprocess.Popen([nohup_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def distributeInfer(conf, date, ps, emrName, logpath=emr_home_path, use_TFoS=True, dirs2issue=None, executer_num=0):
    _args_useSpark = True
    if dirs2issue is None:
        dirs2issue = []
    conf = _find_conf(conf, logpath)
    if __name__=='__main__':
        py_file_conf = _addConfig(conf)
        package_names = ['model', 'utils', 'conf', 'layer']
        py_file_zip = _zipDir(package_names, _project_path)
        py_file = ','.join(py_file_zip + py_file_conf + dirs2issue)
    else:
        py_file_conf = _addConfig(conf)
        py_file = ','.join(py_file_conf + dirs2issue)

    py_file = '--py-files {}'.format(py_file) if py_file != '' else ''
    setEnv()
    if executer_num==0:
        _, _, executer_num = getHostnameList_Chief_Hostnum(emrName)
        if use_TFoS:
            executer_num = int(executer_num) - 1

    use_TFoS_param = '--executor-cores 1' if use_TFoS else ''
    date = '--date {}'.format(date) if date != '' else ''
    command = r'''
    spark-submit --master yarn {py_file} {archive} {TFoS} --num-executors {executor} --executor-memory 15G --driver-memory 15G --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" --conf spark.executorEnv.PYSPARK_PYTHON="/usr/bin/python3" --conf spark.executorEnv.PYSPARK_DRIVER_PYTHON="/usr/bin/python3"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${{CLASSPATH}}"  --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/lib/hadoop/lib/native:${{JAVA_HOME}}/lib/amd64/server" {exec_file} {date} --mode infer --useSpark --logpath {tf_log_path} --ps {ps} --num_executor {num_executor} {use_TFoS} --conf {conf}  > {logpath}
    '''.format(py_file=py_file, archive= '', TFoS=use_TFoS_param, date=date, executor=executer_num, exec_file=os.path.join(_project_path, _exec_file),
               tf_log_path=logpath, ps=ps, num_executor=executer_num, use_TFoS='--use_TFoS' if use_TFoS else '', conf=conf,
               logpath=os.path.join(logpath, 'dl_rank_run.log'))
    process = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    return process.pid

if __name__=='__main__':
    '''
    python3 symphony.py --conf I2Iconf_uv --date 2019-07-24:2019-08-04 --mode train --ps 1 --ev 1
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps', type=int, help='number of parameter server', default=1)
    parser.add_argument('--ev', type=int, help='number of evaluator', default=1)
    parser.add_argument('--emr', type=str, help='name of cluster', default='wangqi')
    parser.add_argument('--keyFile', type=str, help='keyfile path on chief node', default='/home/hadoop/wangqi.pem')
    parser.add_argument('--date', type=str, help='date of dataset', default='')
    parser.add_argument('--conf', type=str, help='select a config in config dir', default='')
    parser.add_argument('--logpath', type=str, help='log dir path', default=emr_home_path)
    parser.add_argument('--mode', type=str, help='train|infer|stop|export|tb', default='train')
    parser.add_argument('--port', type=int, help='default port for tensorboard', default=6006)
    parser.add_argument('--useSpark', help='use spark-file when infer', action='store_true')
    parser.add_argument('--retrain', help='if retrain clear summary', action='store_true')
    parser.add_argument('--use_TFoS', help='use tensorflowonspark when infer', action='store_true')

    _args = parser.parse_args()
    # sth static
    EmrName = _args.emr
    key_filepath = _args.keyFile
    # config of cluster
    ps_num = _args.ps
    ev_num = _args.ev
    _args.logpath = os.path.join(os.getcwd(), _args.logpath)
    if not gfile.Exists(_args.logpath):
        gfile.MakeDirs(_args.logpath)

    if _args.mode == 'train':
        assert _args.conf != '', 'please set conf!'
        distributeTrain(_args.conf, _args.date, _args.retrain,
                        ps_num, ev_num, EmrName, _args.keyFile, _args.logpath, [_project_path])
    elif _args.mode == 'stop':
        stopCluster(EmrName, _args.keyFile)
    elif _args.mode == 'infer':
        distributeInfer(_args.conf, _args.date, ps_num, EmrName, _args.logpath, _args.use_TFoS)
    elif _args.mode == 'tb':
        run_tensorboard(conf=_args.conf, port=_args.port, logpath=_args.logpath)
    elif _args.mode == 'export':
        os.system("python3 {exec_file} --conf {conf} --mode export".format(exec_file=os.path.join(_project_path, _exec_file)
                                                                         , conf=_args.conf))
    elif _args.mode == 'export_online':
        os.system("python3 {exec_file} --conf {conf} --mode export_online".format(exec_file=os.path.join(_project_path, _exec_file)
                                                                         , conf=_args.conf))


