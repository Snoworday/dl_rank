import tensorflow as tf
import os
import shutil
import numpy as np
import time
from collections import OrderedDict


try:
    from ..BaseParser import BaseParser
except:
    from dl_rank import BaseParser
import yaml
import logging

class Parser(BaseParser):

    pass