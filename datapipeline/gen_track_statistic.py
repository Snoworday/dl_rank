from IPython.core.display import display, HTML

import sys
sys.path.append('./python')
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import base64

#-------------------read proto--------------------
path = '/Users/snoworday/Downloads/track'
with open (path+"/stats.pb.txt", "rb") as f:
    content=f.read()
protostr = content.decode("utf-8")

#-------------------gen html----------------------
from IPython.core.display import display, HTML

HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
html = HTML_TEMPLATE.format(protostr=protostr)
# from IPython.core.display import display, HTML

display(HTML(html))