# -*- coding: utf-8 -*-
"""
configuration of iris test
RenMin20191024
"""

class Config(object):
    def __init__(self):
        # image path
        self._root_path = []
        self._train_list = []
        self._num_class = []
        self._normalize = []   
        ####################### training dataset #########################     


        ####################### testing dataset #########################
        # NDCrossSensor
        self._root_path_gallery = ['../../NDCrossSensor/NormIm/']
        self._gallery_list = ['../../NDCrossSensor/LG2200_test_filtered.txt']

        self._root_path_probe = ['../../NDCrossSensor/NormIm/']
        self._probe_list = ['../../NDCrossSensor/LG4000_test_filtered.txt']

        self.data_name = 'Notre Dame CrossSensor2013'
        self.test_type = 'Cross'
        
        
    def num_classGet(self):
        return self._num_class

    def load_detailGet(self):
        return self._root_path, self._train_list

    def gallery_loaderGet(self):
        return  self._root_path_gallery, self._gallery_list
    def probe_loaderGet(self):
        return  self._root_path_probe, self._probe_list