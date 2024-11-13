#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/11/09 17:26:28

# the overall model structure is defined here
from .aqa import AQA as aqa
from .musdl_aqa import MUSDL_AQA as musdl_aqa
from .core_aqa import CORE_AQA as core_aqa

# backbone
from .backbone.i3d import I3D as i3d

# neck
from .neck.avg import AVG as avg

# head
from .head.musdl import MUSDL as musdl
from .head.dae import DAE as dae
from .head.hgcn import HGCN as hgcn
from .head.core import CORE as core
from .head.gdlt import GDLT as gdlt
