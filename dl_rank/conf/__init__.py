
def show_conf(conf_path, conf_type):
    import yaml
    conf_path = _find_conf(conf_path)
    with open(conf_path+'/'+conf_type+'.yaml', 'w') as f:
        type_conf = yaml.load('conf')
    print("Using {} config:".format(conf_type))
    for k, v in type_conf.items():
        print('{}: {}'.format(k, v))



def conf_parser(conf_name, conf_save_path, mode, useSpark):
    import importlib.util
    from . import BaseParser
    if conf_name.startswith('s3'):
        useSpark = False
    absolute_dir, parser_path = _find_conf(conf_name, conf_save_path)
    try:
        spec = importlib.util.spec_from_file_location(conf_name, parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Parser = module.Parser
    except:
        Parser = BaseParser.BaseParser

    return Parser(absolute_dir, mode, useSpark)

def _find_conf(conf_name, conf_save_path):
    from tensorflow import gfile
    import os
    if conf_name.startswith('s3'):
        absolute_dir = conf_name
        gfile.Copy(os.path.join(absolute_dir, 'parser.py'), os.path.join(conf_save_path, 'parser.py'), overwrite=True)
        parser_path = os.path.join(conf_save_path, 'parser.py')
    else:
        if conf_name.startswith('./'):
            conf_name = conf_name[2:]
        absolute_dir = os.path.join(os.getcwd(), conf_name)
        _absolute_dir = absolute_dir
        if not os.path.exists(absolute_dir):
            conf_path = __file__.rsplit('/', 1)[0]
            absolute_dir = os.path.join(conf_path, conf_name)
            absolute_dir = '/'+absolute_dir.strip('/')
        parser_path = os.path.join(absolute_dir, 'parser.py')
    assert gfile.Exists(absolute_dir), 'Cant find Conf directory: {}|{}|{}|{}'.format(absolute_dir, _absolute_dir, conf_name, conf_save_path)
    return absolute_dir, parser_path
