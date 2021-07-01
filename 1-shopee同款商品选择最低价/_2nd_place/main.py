# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import os
from _2nd_place.feature.get_feature import get_image_and_multi_modal_features, get_nlp_features
from _2nd_place.config import Config

__all__ = []

# define class


# define function
def get_feature():
    config = Config(data_path, save_dir)

    get_image_and_multi_modal_features(config, csv_path, image_dir_path, image_model1_ckpt, image_model2_ckpt,
                                       multi_modal_model_ckpt, to_cuda=False, nrows=None)

    get_nlp_features(config, csv_path, image_dir_path, to_cuda, nrows=None)




# main
if __name__ == '__main__':
    pass