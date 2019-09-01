from .wide_and_deep import wdl
from .xcross import xcross
from .xdeepfm import xdeepfm
from .dcn import dcn
from .deepfm_v2 import deepfm as deepfm_v2
from .deepfm import deepfm
from .wide_deep_traditional_attention import wide_and_deep as wide_and_deep_traditional_attention
model_dict = {'wdl': wdl, 'xcross': xcross, 'dcn': dcn, 'xdeepfm': xdeepfm, 'deepfm': deepfm,
              'deepfm_v2': deepfm_v2, 'wide_and_deep_traditional_attention': wide_and_deep_traditional_attention}