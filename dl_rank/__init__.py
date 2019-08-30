from .conf import BaseParser
from . import model
from . import main as _main
from . import distribute as _distribute

__version__ = '0.1.1'

_env = {'dist': True, 'date': '', 'ev_num': 1, 'conf': '', 'logpath': _distribute.emr_home_path, 'backend': False}
_emr = {'keyFile': '/home/hadoop/wangqi.pem', 'emrName': 'wangqi'}

def local():
    global _env
    _env['dist'] = False
    _env['logpath'] = ''

def distribution():
    global _env
    _env['dist'] = True
    _env['logpath'] = _distribute.emr_home_path
    _distribute._setEnv()

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
        import os
        if os._exists(os.path.join(os.getcwd(), keyFile)):
            _emr['keyFile'] = os.path.join(os.getcwd(), keyFile)
        else:
            assert False, 'Cant find keyFile'
    print(_emr)

def set_env(conf=None, date=None, ev_num=None, logpath=None, dist=None, backend=None):
    if conf is not None:
        _env['conf'] = conf
    if date is not None:
        _env['date'] = date
    if ev_num is not None:
        _env['ev_num'] = ev_num
    if logpath is not None:
        _env['logpath'] = logpath
    if dist is not None:
        _env['dist'] = dist
    if backend is not None:
        _env['backend'] = backend

@_update_env
def train(conf, date):
    if _env['dist']:
        _distribute.distributeTrain(conf, date, _emr['ps'], _emr['ev_num'], emrName=_emr['emrName'],
                                    keyFile=_emr['keyFile'], logpath=_env['logpath'])
    else:
        if _env['backend']:
            import subprocess, os
            cmd = 'python3 -m dl_rank.local --mode train --conf {} --useSpark False --date {} --logpath {} > {}'.format(
                conf, date, _env['logpath'], os.path.join(_env['logpath'], 'dl_rank_run.log')
            )
            subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        else:
            _main.run('train', conf=conf, useSpark=False, date=date, logpath=_env['logpath'])


@_update_env
def infer(conf, date):
    if _env['dist']:
        _distribute.distributeInfer(conf, date, logpath=_env['logpath'])
    else:
        if _env['backend']:
            import subprocess, os
            cmd = 'python3 -m dl_rank.local --mode infer --conf {} --useSpark True --date {} --logpath {} > {}'.format(
                conf, date, _env['logpath'], os.path.join(_env['logpath'], 'dl_rank_run.log')
            )
            subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        else:
            _main.run('infer', conf=conf, useSpark=True, date=date, logpath=_env['logpath'])

def stop():
    if _env['dist']:
        _distribute.stopCluster(_emr['emrName'], _emr['keyFile'])

def export(save_path='', conf=''):
    if conf == '':
        conf = _env['conf']
    _main.run('export', conf=conf, useSpark=False, logpath=_env['logpath'], save_path=save_path)

def export4online(conf='', **kwargs):
    if conf == '':
        conf = _env['conf']
    if conf == '' or kwargs:
        _main.EstimatorManager._export_model_online(**kwargs)
    else:
        _main.run('export_online', conf=conf, **kwargs)

def start_tensorboard(path='', conf=''):
    if path != '':
        _distribute.run_tensorboard(model_dir=path, logpath=_env['logpath'])
    elif conf != '':
        _distribute.run_tensorboard(conf=conf, logpath=_env['logpath'])
    elif _env['conf'] != '':
        _distribute.run_tensorboard(conf=_env['conf'], logpath=_env['logpath'])
    else:
        raise FileNotFoundError('Please define summary path')
