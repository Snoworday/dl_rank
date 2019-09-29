import yaml


class ConfigFactory(object):
    def __init__(self, **template):
        self.feature = template['feature']
        self.mission = template['mission']
        self.model = template['model']
        self.schema = template['schema']
        self.vocabulary = template['vocabulary']



