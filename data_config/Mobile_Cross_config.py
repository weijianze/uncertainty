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
        # MobileV2_data
        self._root_path_test = ['../../MobileV2/']
        self._test_list = ['../../MobileV2/test.txt']

        self.data_name = 'CASIA Mobile'
        self.test_type = 'Cross'
        
    def num_classGet(self):
        return self._num_class

    def load_detailGet(self):
        return self._root_path, self._train_list
    
    def test_loaderGet(self):
        return  self._root_path_test, self._test_list
