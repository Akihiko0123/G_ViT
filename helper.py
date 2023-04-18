#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
from PIL import Image
import os

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    adj_s = [ b['adj_s'] for b in batch ]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}

# 下記Trainクラスのtrainerメソッドから呼び出し
# DataLoaderを通したバッチごとの情報(元はWSI単位の情報)を引数で受け取る
# batch_graph=特徴データ、batch_label=正解ラベル(1or0等)、batch_adjs=グラフデータ(隣接かどうか)
def preparefeatureLabel(batch_graph, batch_label, batch_adjs):
    # 0次元目方向の長さは1WSI中のバッチの数と思われる
    batch_size = len(batch_graph)
    # バッチサイズ(バッチの数？)だけ項目を持つ値0の64bit float tensorを用意する。
    # 例：バッチサイズ10の場合,tensor([0,0,0,0,0,0,0,0,0,0])
    labels = torch.LongTensor(batch_size)
    max_node_num = 0
    
    # バッチの数だけ繰り返し(例：batch_size = 10 =>10回)
    for i in range(batch_size):
        # iバッチ目のラベル(0or1[or2]?)にバッチラベルのi番目を代入する(0or1[or2])
        labels[i] = batch_label[i]
        # ノードの値として、
        # max_node_num = 0か特徴値のパッチ数？のどちらか大きい方(WSI内の全パッチ数？)
        # チェック用
        print(f"helper.py_batch_graph[i].shape:{batch_graph[i].shape}")
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    masks = torch.zeros(batch_size, max_node_num)
    adjs =  torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, 512)

    for i in range(batch_size):
        cur_node_num =  batch_graph[i].shape[0]
        #node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        #adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        
        #masks
        masks[i,0:cur_node_num] = 1  
    # cpu対応できるように変更
    if torch.cuda.is_available():
        node_feat = batch_node_feat.cuda()
        labels = labels.cuda()
        adjs = adjs.cuda()
        masks = masks.cuda()
    else:
        node_feat = batch_node_feat
        labels = labels
        adjs = adjs
        masks = masks

    return node_feat, labels, adjs, masks

# main.pyからクラス数の引数(n_class)で呼び出される
class Trainer(object):
    def __init__(self, n_class):
        # 混同行列,utilsフォルダ内、metrics.pyのConfusionMatrixクラスを呼び出し
        # metrics = ConfusionMatrixクラスのインスタンス
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()
    
    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model):
        # 特徴ラベルの準備を行う上記のpreparefeatureLabel関数を呼び出し
        # sample['label'] = ラベル(0,1等)
        # sample['id'] = ptファイルのフォルダ名？(output等)
        # sample['image'] = 特徴データ
        # sample['adj_s'] = グラフ情報のテンソル
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'])
        pred,labels,loss = model.forward(node_feat, labels, adjs, masks)

        return pred,labels,loss

class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def eval_test(self, sample, model, graphcam_flag=False):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'])
        if not graphcam_flag:
            with torch.no_grad():
                pred,labels,loss = model.forward(node_feat, labels, adjs, masks)
        else:
            torch.set_grad_enabled(True)
            pred,labels,loss= model.forward(node_feat, labels, adjs, masks, graphcam_flag=graphcam_flag)
        return pred,labels,loss