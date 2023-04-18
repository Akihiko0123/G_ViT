#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options
# LinuxかWindowsか判別用
import platform

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init


# 同階層のoption.pyのクラスを用いて引数を取り出す
args = Options().parse()
# 引数からクラス数を設定
n_class = args.n_class

# CUDA上の全てのカーネルが準備完了するのを待つ。
# CUDAを用いる場合は、これにより実行時間を正確に計測することができる。
# cpuに対応させる
if torch.cuda.is_available():
    torch.cuda.synchronize()
# seed固定のように、Pytorchにて再現性を実現するオプション。
# 設定しない場合と比べて速度は落ちるので注意
torch.backends.cudnn.deterministic = True

# 引数からデータのパスを取得、train.shによるとグラフデータとのことなので
# テンソルマップである.ptファイルのパスを指定すると思われたが、
# 後のコードを見ると怪しい。最後にptファイルが入っているgraphsディレクトリのパスが正しいように思う
data_path = args.data_path
# 引数からモデル出力用のパスを取得
model_path = args.model_path
print(f"[main.py]model_path:{model_path}")
# モデル出力用のパス(model_path)がディレクトリでない場合(モデルパス名のディレクトリがない場合を含む)は、
# ディレクトリを作成する
if not os.path.isdir(model_path): os.mkdir(model_path)
# 引数からログファイル用のパスを取得
log_path = args.log_path
# ログファイル用のパス名のディレクトリがない場合は作成する
if not os.path.isdir(log_path): os.mkdir(log_path)
# 引数からタスク名を取得（モデル名やログ名に使用）
task_name = args.task_name

print(task_name) # タスク名を表示
###################################
train = args.train # 引数からtrainを取得(True or False)
test = args.test   # 引数からtestを取得(True or False)
graphcam = args.graphcam # 引数からgraphcamを取得(True or False)
print("train:", train, "test:", test, "graphcam:", graphcam) # 引数の情報を表示

##### Load datasets
print("preparing datasets and dataloaders......")
# 引数からバッチサイズの情報を取得
batch_size = args.batch_size

# trainが指定されている場合
if train:
    # train_setは各行に
    #「対照学習モデルを通して出力された特徴ファイルの2つ上のディレクトリ名\\1つ上のディレクトリ名」 -tab- 「ラベル名」
    # が書かれていると思われる
    ids_train = open(args.train_set).readlines() # ids_train = train_set引数で指定したファイルの1行目
    # チェック用
    print(f"[main.py]ids_train:{ids_train}")
    # train.shで実行されるときはtrain_set引数としてtrain.txtが指定されている
    # dataset_train = data_pathのパス+"/or\\"とids_trainを引数にした,
    #                 GraphDatasetクラス(utilsディレクトリ内dataset.py)のインスタンス
    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
    print(f"[main.py]len(dataset_train):{len(dataset_train)}")
    # データセットをバッチサイズに合わせたデータローダに変換
    # DataLoaderは内部的にdataset_trainについてGraphDatasetクラスの__getitem__メソッドを実行しているので、
    # それぞれデータは以下のような特徴とラベルとグラフテンソルの情報を持つ
    # sample['label'] = ラベル(0,1等)
    # sample['id'] = ptファイルのフォルダ名？(output等)
    # sample['image'] = 特徴データ
    # sample['adj_s'] = グラフ情報のテンソル    
    if torch.cuda.is_available():
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=0, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    # バッチ数と１バッチの大きさの積 = 合計学習数 = total_train_num
    total_train_num = len(dataloader_train) * batch_size
    # チェック用
    print(f"[main.py]len(dataloader_train):{len(dataloader_train)}")

# val_set.txtを読み込む
ids_val = open(args.val_set).readlines()
# train_set.txtと同じようにデータローダーを作成
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
if torch.cuda.is_available():
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
else:
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=0, collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### creating models #############
print("creating models......")
#　引数からエポック数を設定
num_epochs = args.num_epochs
# 引数から学習率を設定
learning_rate = args.lr
# model = modelsフォルダ内、GraphTransformer.pyのClassifierクラスのインスタンス
model = Classifier(n_class)
model = nn.DataParallel(model)

if args.resume:
    print('load model{}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.is_available():
    model = model.cuda()
#model.apply(weight_init)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)       # best:5e-4, 4e-3
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

##################################
# 交差エントロピー誤差
criterion = nn.CrossEntropyLoss()

if not test:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

# trainer = helper.pyのTrainerクラスのインスタンス
trainer = Trainer(n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
for epoch in range(num_epochs):
    # optimizer.zero_grad()
    model.train()
    train_loss = 0.
    total = 0.

    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch+1, current_lr, best_pred))

    if train:
        print("train")
        for i_batch, sample_batched in enumerate(dataloader_train):
            # チェック用
            print(f"[main.py]sample_batched:{sample_batched}")
            #scheduler(optimizer, i_batch, epoch, best_pred)
            scheduler.step(epoch)
            # 上記作成済みのhelper.pyのTrainerクラスのインスタンスで、trainメソッドを実行する
            preds,labels,loss = trainer.train(sample_batched, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            total += len(labels)

            trainer.metrics.update(labels, preds)
            #trainer.plot_cm()
            # チェック用
            print("[main.py]total_train_num:%d" %(total_train_num))
            print("[main.py]total:%d" %(total))
            print("[main.py]train_loss:%3f" %(train_loss))

            if (i_batch + 1) % args.log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                trainer.plot_cm()

    if not test: 
        # チェック用
        print("[main.py]total_train_num:%d" %(total_train_num))
        print("[main.py]total:%d" %(total))
        print("[main.py]train_loss:%3f" %(train_loss))
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        trainer.plot_cm()


    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            total = 0.
            batch_idx = 0

            for i_batch, sample_batched in enumerate(dataloader_val):
                # hilper.pyのEvaluatorクラスのインスタンスevaluatorのeval_testメソッドを実行
                #pred, label, _ = evaluator.eval_test(sample_batched, model)
                preds, labels, _ = evaluator.eval_test(sample_batched, model, graphcam)
                
                total += len(labels)

                evaluator.metrics.update(labels, preds)

                if (i_batch + 1) % args.log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    evaluator.plot_cm()

            print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
            evaluator.plot_cm()

            # torch.cuda.empty_cache()

            val_acc = evaluator.get_scores()
            if val_acc > best_pred: 
                best_pred = val_acc
                if not test:
                    print("saving model...")
                    torch.save(model.state_dict(), model_path + task_name + ".pth")

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"

            log += "================================\n"
            print(log)
            if test: break

            f_log.write(log)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)

    trainer.reset_metrics()
    evaluator.reset_metrics()

if not test: f_log.close()