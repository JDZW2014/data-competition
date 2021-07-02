# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2021-07-01
Description :
auther : wcy
"""
# import modules
import os
import torch

__all__ = []

# define function
# checkpoint1 = torch.load('v45.pth', map_location=torch.device('cpu'))
# checkpoint2 = torch.load('v34.pth', map_location=torch.device('cpu'))
# checkpoint3 = torch.load('v79.pth', map_location=torch.device('cpu'))
# params1 = checkpoint1['params']
# params2 = checkpoint2['params']
# params3 = checkpoint3['params']


checkpoint = torch.load('v75.pth', map_location=torch.device('cpu'))
checkpoint2 = torch.load('v102.pth', map_location=torch.device('cpu'))
checkpoint3 = torch.load('v103.pth', map_location=torch.device('cpu'))

params_bert = checkpoint['params']
params_bert2 = checkpoint2['params']
params_bert3 = checkpoint3['params']
pass