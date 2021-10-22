#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import prepare_target_domain
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))
prepare_target_domain(['DataSet/Kaggle/train', 'DataSet/Kaggle/test'],
                      '(.*).jpeg', 'DataSet/Pooled/targets.h5')
