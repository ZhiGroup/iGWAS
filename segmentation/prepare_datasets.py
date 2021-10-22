#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import prepare_datasets
from glob import glob
import os


dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)
datasets = {'aria': (glob('DataSet/ARIA/*markups'),
                     glob('DataSet/ARIA/*vessel'),
                     '(.*).tif',
                     '(.+)_(BSS|BDP)'),
            'chasedb': (['DataSet/CHASEDB1'],
                        ['DataSet/CHASEDB1'],
                        '(.*).jpg',
                        '(.+)_[12]stHO'),
            'drive_train': (['DataSet/DRIVE/training/images'],
                            ['DataSet/DRIVE/training/1st_manual'],
                             '(.*)_training.tif',
                             '(.*)_manual1'),
            'drive_test': (['DataSet/DRIVE/test/images'],
                           ['DataSet/DRIVE/test/1st_manual',
                            'DataSet/DRIVE/test/2nd_manual'],
                            '(.*)_test.tif',
                            '(.*)_manual[12]'),
            'hrf': (['DataSet/HRF/images'],
                    ['DataSet/HRF/manual1'],
                    '(.*).(jpg|JPG)',
                    '(.*).tif'),
            'iostar': (['DataSet/IOSTAR/image'],
                       ['DataSet/IOSTAR/GT'],
                       '(.*).jpg',
                       '(.*)_GT.tif'),
            'stare': (['DataSet/STARE/images'],
                      ['DataSet/STARE/manual1',
                       'DataSet/STARE/manual2'],
                       '(.*).ppm',
                       '(.*).(ah|vk).ppm')
            }
target_hdf5 = 'DataSet/Pooled/datasets.h5'

prepare_datasets(datasets, target_hdf5)
