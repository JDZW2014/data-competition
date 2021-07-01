# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-06-30
Description :
auther : wcy
"""
# import modules
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import torch
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
from PIL import Image


__all__ = ["ShopeeDataset", "load_data", "gem", "BertDataset", "image_transformer"]


# define class
class ShopeeDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = read_image(str(self.img_dir / row['image']))
        _, h, w = img.shape
        st_size = (self.img_dir / row['image']).stat().st_size
        if self.transform is not None:
            img = self.transform(img)

        return img, row['title'], h, w, st_size

    def __len__(self):
        return len(self.df)


class BertDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if 'y' in row.keys():
            target = torch.tensor(row['y'], dtype=torch.long)
            return row['title'], target
        else:
            return row['title']

    def __len__(self):
        return len(self.df)


# define function
def load_data(csv_path, image_dir_path, nrows=None):

    if nrows is None:
        df = pd.read_csv(csv_path, usecols=['posting_id', 'image', 'title'])
    else:
        df = pd.read_csv(csv_path, nrows=nrows, usecols=['posting_id', 'image', 'title'])

    img_dir = Path(image_dir_path)

    return df, img_dir


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


def image_transformer(image_size, expand_size=32,
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), interpolation=Image.BICUBIC):
    return Compose([
        Resize(size=image_size + expand_size, interpolation=interpolation),
        CenterCrop((image_size, image_size)),
        Normalize(mean=mean, std=std),
    ])

