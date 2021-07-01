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
import logging

logger = logging.getLogger()

__all__ = []

# define class


# define function
def get_feature_from_pretrained_model(logger, config):

    image_model1_ckpt = ''
    image_model2_ckpt = ''
    multi_modal_model_ckpt = ''
    get_image_and_multi_modal_features(logger=logger, config=config, image_model1_ckpt=image_model1_ckpt,
                                       image_model2_ckpt=image_model2_ckpt,
                                       multi_modal_model_ckpt=multi_modal_model_ckpt, to_cuda=False, nrows=None)
    bert_model_ckpt = ''
    bert2_model_ckpt = ''
    bert3_model_ckpt = ''

    get_nlp_features(logger=logger, config=config, bert_model_ckpt=bert_model_ckpt, bert2_model_ckpt=bert2_model_ckpt,
                     bert3_model_ckpt=bert3_model_ckpt, nrows=None)


def main():
    config = Config(data_path="", image_dir_path="", save_dir="temp")
    get_feature_from_pretrained_model(config=config, logger=logger)


# main
if __name__ == '__main__':
    main()
