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
        # CASIA-Thousand
        self._root_path_test = ['../../CASIA-Iris-Thousand/']
        self._test_list = ['../../CASIA-Iris-Thousand/test.txt']

        self.data_name = 'CASIA IrisV4-thousand'
        self.test_type = 'Cross'

    def num_classGet(self):
        return self._num_class

    def load_detailGet(self):
        return self._root_path, self._train_list

    def test_loaderGet(self):
        return  self._root_path_test, self._test_list
