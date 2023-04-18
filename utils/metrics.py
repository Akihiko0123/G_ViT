# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

# main.pyから呼び出されたhelper.py内Trainerクラスのコンストラクタから呼び出される
# 恐らく混同行列を作成するクラス
class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: prediction
        # axis = 1: target
        # 混同行列の枠として、クラス数×クラス数のnumpy配列を用意する
        # 0次元を予測値, 1次元を正解とする。
        # 例：2クラスの場合、
        #          正  解
        #         陽性 陰性
        # 予 陽性 [ 0,  0 ]
        # 測 陰性 [ 0,  0 ] 
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    # 
    def _fast_hist(self, label_true, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        hist[label_pred, label_true] += 1

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.item(), lp.item(), self.n_classes)    #lt.item(), lp.item()
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along

        if sum(hist.sum(axis=1)) != 0:
            acc = sum(np.diag(hist)) / sum(hist.sum(axis=1))
        else:
            acc = 0.0
        
        return acc
    
    def plotcm(self):
        print(self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))