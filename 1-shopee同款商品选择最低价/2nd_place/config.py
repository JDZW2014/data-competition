# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
import os
import pandas as pd
from pathlib import Path

__all__ = []

# config
k = 50
conf_th = 0.7

DEBUG = len(pd.read_csv('../input/shopee-product-matching/test.csv')) == 3


# define function
def load_data():
    if DEBUG:
        nrows = 1000
        df = pd.read_csv('../input/shopee-product-matching/train.csv', nrows=nrows,
                         usecols=['posting_id', 'image', 'title'])
        # nrows = None
        # df = pd.read_csv('../input/shopee-product-matching/train.csv', nrows=nrows, usecols=['posting_id', 'image', 'title']).append(
        # pd.read_csv('../input/shopee-product-matching/train.csv', nrows=nrows, usecols=['posting_id', 'image', 'title'])).reset_index(drop=True)
        img_dir = Path('../input/shopee-product-matching/train_images/')
    else:
        nrows = None
        df = pd.read_csv('../input/shopee-product-matching/test.csv', usecols=['posting_id', 'image', 'title'])
        img_dir = Path('../input/shopee-product-matching/test_images/')
    return df, img_dir


# main
if __name__ == '__main__':
    pass
