from .conf.BaseParser import BaseParser
from . import model
from . import solo as _main
from . import conf as _conf
from . import symphony as _distribute
import os as _os
import signal as _signal
import subprocess as _subprocess

__version__ = '0.1.12'

_env = {'distribution': True, 'date': '', 'ev_num': 1, 'ps_num': 1, 'conf': '', 'tb_port': 6006, 'use_TFoS': False,
        'save_data_type': 'txt', 'logpath': _distribute.emr_home_path, 'backend': False, 'force_stop': True}
_emr = {'keyFile': '/home/hadoop/wangqi.pem', 'emrName': 'wangqi'}
_process_pool = {'train': None, 'infer': None, 'tb': None}

def emr():
    for k, v in _emr.items():
        print(k, ':', v)

def env():
    for k, v in _env.items():
        print(k, ':', v)

def model_conf():
    _conf.show_conf(_env['conf'], 'model')

def train_conf():
    _conf.show_conf(_env['conf'], 'train')

def local():
    global _env
    _env['distribution'] = False
    _env['logpath'] = _os.getcwd()

def distribution():
    global _env
    _env['distribution'] = True
    _env['logpath'] = _distribute.emr_home_path

def backend(on=True):
    _env['backend'] = on

def off_eval():
    _env['ev_num'] = 0

def _update_env(mode_fn):
    def wrapper(conf=None, date=None):
        if conf is not None:
            _env['conf']=conf
        else:
            try:
                conf = _env['conf']
            except KeyError:
                assert False, 'Configure Dict should be set'
        if date is not None:
            _env['date'] = date
        else:
            date = _env['date']
        mode_fn(conf, date)
    return wrapper

def set_emr(emrName=None, keyFile=None):
    if emrName is not None:
        _emr['emrName'] = emrName
    if keyFile is not None:
        if _os._exists(_os.path.join(_os.getcwd(), keyFile)):
            _emr['keyFile'] = _os.path.join(_os.getcwd(), keyFile)
        else:
            assert False, 'Cant find keyFile'
    print(_emr)

def set_env(conf=None, date=None, ev_num=None, ps_num=None, use_TFoS=None, logpath=None, distribution=None,
            backend=None, force_stop=None, save_data_type=None):
    if conf is not None:
        _env['conf'] = conf
    if date is not None:
        _env['date'] = date
    if ev_num is not None:
        if ev_num > 1:
            print('evaluator number can only be 1 now')
            ev_num = 1
        _env['ev_num'] = ev_num
    if ps_num is not None:
        _env['ps_num'] = ps_num
    if use_TFoS is not None:
        _env['use_TFoS'] = use_TFoS
    if logpath is not None:
        _env['logpath'] = _os.path.join(_os.getcwd(), logpath)
    if distribution is not None:
        _env['distribution'] = distribution
    if backend is not None:
        _env['backend'] = backend
    if force_stop is not None:
        _env['force_stop'] = force_stop
    if save_data_type is not None:
        assert save_data_type in ['ckpt', 'npy', 'txt'], 'savetype of variables should be one of ckpt|npy|txt'
        _env['save_data_type'] = save_data_type

@_update_env
def train(conf, date):
    if _env['distribution']:
        process = _distribute.distributeTrain(conf, date, _env['ps_num'], _env['ev_num'], emrName=_emr['emrName'],
                                              keyFile=_emr['keyFile'], logDir=_env['logpath'])
        print('Train Process pid: {}'.format(process.pid))
    else:
        if _env['backend']:
            if _process_pool['train'] is not None:
                _kill_process(_process_pool['train'])
            date = '' if date=='' else '--date {}'.format(date)
            cmd = 'python3 -m dl_rank.solo --mode train --conf {} {} --logpath {} --ps {} > {}'.format(
                conf, date, _env['logpath'], _env['ps_num'],
                _os.path.join(_env['logpath'], 'dl_rank_run.log')
            )
            process = _subprocess.Popen([cmd], shell=True, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, close_fds=True)
            _process_pool['train'] = process.pid
            print('Train Process pid: {}'.format(process.pid))
        else:
            _main.run('train', conf=conf, useSpark=False, date=date, ps=_env['ps_num'],
                      logpath=_env['logpath'])



@_update_env
def test(conf, date):
    pred_generator = _main.run('test', conf=conf, useSpark=False, date=date, ps=True,
              logpath=_env['logpath'])
    return pred_generator

@_update_env
def infer(conf, date):
    if _env['distribution']:
        process = _distribute.distributeInfer(conf, date,_env['ps_num'], _emr['emrName'],
                                              logpath=_env['logpath'], use_TFoS=_env['use_TFoS'])
        print('Infer Process pid: {}'.format(process.pid))
    else:
        if _env['backend']:
            if _process_pool['infer'] is not None:
                _kill_process(_process_pool['infer'])
            date = '' if date=='' else '--date {}'.format(date)
            cmd = 'python3 -m dl_rank.solo --mode infer --conf {} {} --logpath {} --ps {} > {}'.format(
                conf, date, _env['logpath'], _env['ps_num'],
                _os.path.join(_env['logpath'], 'dl_rank_run.log')
            )
            process = _subprocess.Popen([cmd], shell=True, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, close_fds=True)
            _process_pool['infer'] = process.pid
            print('Infer Process pid: {}'.format(process.pid))
        else:
            _main.run('infer', conf=conf, useSpark=False, date=date, ps=_env['ps_num'],
                      logpath=_env['logpath'])

def stop(task=None):
    if task=='train' or task==None:
        if _env['distribution']:
            _distribute.stopCluster(_emr['emrName'], _emr['keyFile'])
            _process_pool['train'] = None
        else:
            _kill_process('train')
    elif task=='infer' or task==None:
        _kill_process('infer')
    elif task=='tb' or task==None:
        _kill_process('tb')

def export(save_path=None, conf='', pbtxt=True):
    if conf == '':
        conf = _env['conf']
    _main.run('export', conf=conf, useSpark=False, logpath=_env['logpath'], save_path=save_path, pbtxt=pbtxt)

def export4online(conf='', from_pb=False, **kwargs):
    if conf == '':
        conf = _env['conf']
    if conf == '' or kwargs:
        _main.EstimatorManager._export_model_online(**kwargs)
    else:
        _main.run('export_online', conf=conf, useSpark=False, output_data_type=_env['save_data_type'],
                  from_pb=from_pb, logpath=_env['logpath'])

def start_tensorboard(path='', conf='', port=None):
    if port is not None:
        _env['tb_port'] = port
    if _process_pool['tb'] is None:
        _kill_process(_process_pool['tb'])
        if path != '':
            process = _distribute.run_tensorboard(model_dir=path, logpath=_env['logpath'], port=_env['tb_port'])
        elif conf != '':
            process = _distribute.run_tensorboard(conf=conf, logpath=_env['logpath'], port=_env['tb_port'])
        elif _env['conf'] != '':
            process = _distribute.run_tensorboard(conf=_env['conf'], logpath=_env['logpath'], port=_env['tb_port'])
        else:
            raise FileNotFoundError('Please define summary path')
        _process_pool['tb'] = process.pid
    else:
        pass
    print('Tensorboard Process pid: {}, port: {}'.format(_process_pool['tb'], _env['tb_port']))


def _kill_process(group_name=None):
    group_names = ['tb', 'infer', 'train'] if group_name is None else [group_name]
    for group_name in group_names:
        if group_name == 'tb':
            _os.system("ps -ef | grep python | grep {name} | awk '{{print \"kill -9 \" $2}}' | bash -v"
                       .format(name='tensorboard'))
            _process_pool[group_name] = None
        else:
            if _process_pool[group_name] is None and _env['force_stop']:
                _os.system("ps -ef | grep python | grep solo | grep {name} | awk '{{print \"kill -9 \" $2}}' | bash -v"
                       .format(name=group_name))
            else:
                import psutil
                root = psutil.Process(_process_pool[group_name])
                childs = root.children(recursive=True)
                for chd in childs:
                    chd.kill()
                root.kill()
                _process_pool[group_name] = None


