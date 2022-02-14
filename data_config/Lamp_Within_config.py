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
        CASIA-Lamp
        self._root_path.append('../../CASIA-Iris-Lamp/')
        self._train_list.append('../../CASIA-Iris-Lamp/train.txt')
        self._num_class.append(410)


        ####################### testing dataset #########################
        # CASIA-Lamp
        self._root_path_test = ['../../CASIA-Iris-Lamp/']
        self._test_list = ['../../CASIA-Iris-Lamp/test.txt']
        
        self.data_name = 'CASIA IrisV4-lamp'
        self.test_type = 'Within'
        
    def num_classGet(self):
        return self._num_class

    def load_detailGet(self):
        return self._root_path, self._train_list
    
    def test_loaderGet(self):
        return  self._root_path_test, self._test_list
