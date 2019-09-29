from .BaseModel import baseModel
from .wide_and_deep import wdl
from .xcross import xcross
from .xdeepfm import xdeepfm
from .dcn import dcn
from .deepfm_v2 import deepfm as deepfm_v2
from .deepfm import deepfm
from .wide_deep_traditional_attention import wide_and_deep as wide_and_deep_traditional_attention

import os
import importlib.util


class modelFactory(object):
    model_dict = {'wdl': wdl, 'xcross': xcross, 'dcn': dcn, 'xdeepfm': xdeepfm, 'deepfm': deepfm,
                  'deepfm_v2': deepfm_v2, 'wide_and_deep_traditional_attention': wide_and_deep_traditional_attention}
    external_path = None
    @staticmethod
    def build(train_conf, model_conf, mode):
        model_type = train_conf['model_type']
        if model_type not in modelFactory.model_dict:
            exter_module_path = os.path.join(modelFactory.external_path, model_type)
            if os.path.exists(exter_module_path):
                exter_spec = importlib.util.spec_from_file_location('dl_rank.model.'+model_type, exter_module_path)
                exter_module = importlib.util.module_from_spec(exter_spec)
                exter_spec.loader.exec_module(exter_module)
                modelFactory._find_submodel(exter_module)
            else:
                assert False, 'Cant find external model: {}'.format(exter_module_path)
        model = modelFactory.model_dict[model_type](model_conf, mode)
        return model

    @staticmethod
    def register(model):
        name = model.name if hasattr(model, 'name') else model.__name__
        if name in modelFactory.model_dict:
            print('name: {} has been used'.format(name))
        else:
            modelFactory.model_dict.update({name: model})

    @staticmethod
    def _find_submodel(module):
        for attr in dir(module):
            attrObj = getattr(module, attr)
            if issubclass(attrObj, baseModel) and not isinstance(attrObj, baseModel):
                modelFactory.register(attrObj)


