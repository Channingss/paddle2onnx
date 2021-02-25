# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import yaml
from collections import OrderedDict

from .ppdet_config_parse import parse_ppdet_config


class ConfigParser(object):
    def __init__(self, cfg_file, pp_type):
        if not osp.exists(cfg_file):
            raise Exception(
                "cfg_file {} does not exist, please set the right file.".format(
                    cfg_file))
        if pp_type not in ['det']:
            raise Exception(
                "pp_type should be one of det/, but recieved type is {}".format(
                    pp_type))
        if pp_type == 'det':
            self.config = parse_ppdet_config(cfg_file)
