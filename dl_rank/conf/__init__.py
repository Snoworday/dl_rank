def conf_parser(conf_name, useSpark):
    import importlib.util
    from . import BaseParser
    import os

    conf_path = __file__.rsplit('/', 1)[0]

    absolute_dir = os.path.join(os.getcwd(), conf_name)
    if not os._exists(absolute_dir):
        absolute_dir = os.path.join(conf_path, conf_name)
    try:
        spec = importlib.util.spec_from_file_location(conf_name, os.path.join(absolute_dir, 'parser.py'))
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        Parser = foo.Parser
    except:
        Parser = BaseParser.BaseParser

    return Parser(conf_name, useSpark)

