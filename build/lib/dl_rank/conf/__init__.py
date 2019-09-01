
def show_conf(conf_path, conf_type):
    import yaml
    conf_path = _find_conf(conf_path)
    with open(conf_path+'/'+conf_type+'.yaml', 'w') as f:
        type_conf = yaml.load('conf')
    print("Using {} config:".format(conf_type))
    for k, v in type_conf.items():
        print('{}: {}'.format(k, v))



def conf_parser(conf_name, useSpark):
    import importlib.util
    from . import BaseParser
    import os
    absolute_dir = _find_conf(conf_name)
    try:
        spec = importlib.util.spec_from_file_location(conf_name, os.path.join(absolute_dir, 'parser.py'))
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        Parser = foo.Parser
    except:
        Parser = BaseParser.BaseParser

    return Parser(absolute_dir, useSpark)

def _find_conf(conf_name):
    import os
    absolute_dir = os.path.join(os.getcwd(), conf_name)
    if not os.path.exists(absolute_dir):
        conf_path = __file__.rsplit('/', 1)[0]
        absolute_dir = os.path.join(conf_path, conf_name)
    assert os.path.exists(absolute_dir), 'Cant find Conf directory'
    return '/'+absolute_dir.strip('/')
