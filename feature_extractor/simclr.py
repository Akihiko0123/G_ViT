import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
# 4.1テスト用に追加
import matplotlib.pyplot as plt
import torchvision
import numpy as np

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)

# 以下trainメソッドより呼び出される
def _save_config_file(model_checkpoints_folder):
    # モデルチェックポイントのフォルダが無い場合、作成する
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        # 作成したモデルチェックポイントフォルダにconfig.yamlをコピーする
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

# run.pyから呼び出される
class SimCLR(object):

    def __init__(self, dataset, config):
        # run.pyで指定したconfig.yamlよりconfigを読み込み
        self.config = config
        # 以下の_get_deviceメソッドでcudaかcpuか等決定
        self.device = self._get_device()
        # テンソルボードのSummaryWriter機能でテンソルボードを操作、ログ保存用のファイルディレクトリを作成
        # https://rightcode.co.jp/blog/information-technology/tensorboard-in-colaboratory-notebook-pytorch-machine-learning-process-visualize
        # を参照
        # 保存先はデフォルトで./runsになる
        self.writer = SummaryWriter()
        # データセット
        self.dataset = dataset
        # deviceとバッチサイズ、ロス方式(デフォルトでコサイン類似度も使用)を決定
        # nt_xent_criterion = XTXentLossクラスのインスタンス
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device
    # trainメソッドより呼び出される
    def _step(self, model, xis, xjs, n_iter):
        # xisをモデルに入れて、特徴と予測を取り出す
        # model()とすることで、内部で__call__メソッド(_call_impl)からResNetSimCLRクラスのforwardメソッドが呼ばれるので、
        # ris= resnet_children出力後、線形層L1とrelu関数とL2を通す前の層で出力される特徴量
        # zis= resnet_children出力後、線形層L1とrelu関数とL2を通して指定サイズに合わせて出力された特徴量(予測結果？)
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # 予測特徴ベクトル(zisとzjs)を標準化
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # 異なる拡張を施した自身同士の類似度を計算していると思われる
        # loss = NTXentLossクラスのforwardメソッドで計算したロス(類似度[コサイン類似度など])
        loss = self.nt_xent_criterion(zis, zjs)
        return loss
    # run.pyよりsimclrインスタンスが実行
    def train(self):
        # DatasetWrapperクラスのインスタンスであるdatasetのget_data_loadersメソッドを実行し、学習用と検証用のデータローダーを取得
        train_loader, valid_loader = self.dataset.get_data_loaders()

        # チェック用
#        Iter = iter(train_loader)    
#        x_d,y_d = next(Iter)    
#        print("x_d:",x_d)
#        print("y_d:",y_d)
#        for tl in train_loader:
#            print("tl:",tl)

        # モデルはResNetを使用。model = feature_extractor内のmodels内のresnet_simclr.pyにあるResNetSimCLRクラスのインスタンス
        model = ResNetSimCLR(**self.config["model"])# .to(self.device)
        # gpuが1個より多いなら並列処理
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model, device_ids=eval(self.config['gpu_ids']))
        # 事前学習重みを読み込むメソッドを呼び出す。
        # ただし、feature_extractorのruns内の指定箇所のcheckpointディレクトリ内に
        # 事前学習済みのmodel.pthファイルが保存されている場合のみ
        # 保存されていない場合は、事前学習なしで重みを1から計算する。
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)

        print(f"evalで10e-6を処理:{eval(self.config['weight_decay'])}")

        # 最適化optimizer = モデルパラメータ, 学習率(1e-5), 重み減衰(過学習対策) = eval(指定した10e-6を使用)
        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
#                                                                last_epoch=-1)
        # epoch毎に学習率を更新するなどのスケジューラ、T_maxで最小の学習率に達するまでのエポック数(config.yamlのconfig['epochs'])を指定
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)
        
        # fp16_precisionがTrueでapex_supportがTrueの場合に実行
        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)
        # model_checkpoints_folder = writerログのcheckpointsディレクトリパス
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config fileのメソッドを呼び出し、configファイルをチェックポイントフォルダにコピー
        _save_config_file(model_checkpoints_folder)

        # iter数= 0
        n_iter = 0
        # 検証iter数 = 0
        valid_n_iter = 0
        # ベスト検証ロス = 無限 初期値なので大きくしている
        best_valid_loss = np.inf
        
        # エポック数だけ繰り返し
        for epoch_counter in range(self.config['epochs']):
            # 最適化対象の全てのパラメータの勾配初期化
            # 2バッチ(xis, xjs)ずつ実行？
            for (xis, xjs) in train_loader:
                # テスト用4.1追加
#                plt.imshow(np.transpose(torchvision.utils.make_grid(xis).numpy(),(1,2,0))) 
#                plt.imshow(np.transpose(torchvision.utils.make_grid(xjs).numpy(),(1,2,0))) 
#                plt.show()             
#                plt.clf()
#                plt.close()
                # テンソル形式なのでnumpy変換してみる
#                plt.imshow(xis.numpy())
#                plt.imshow(xjs.numpy())
                # テスト用4.1追加は以上
#                print(f"xis,xjs:{(xis,xjs)}")
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # stepメソッドを呼び出しロスを計算
                # ResNetSimCLRクラスのモデルを使用
                # ロスは類似度ベースのクロスエントロピー誤差によるもの
                loss = self._step(model, xis, xjs, n_iter)
                # config.yamlで設定した間隔でログに学習ロスとn_iterを保存
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    # 現行エポック番号と全体エポック数を表示、n_iterと学習ロスも表示する
                    print("[%d/%d] step: %d train_loss: %.3f" % (epoch_counter, self.config['epochs'], n_iter, loss))
                # apex_supportがTrueの場合で、config.yamlでfp16_precisionがTrueの場合に実行(デフォルトはFalse)
                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        # 重み付き逆伝播？
                        scaled_loss.backward()
                # デフォルトの場合、通常通りの逆伝播
                else:
                    loss.backward()

                # 最適化関数によるパラメータを更新
                optimizer.step()
                # n_iterに1追加
                n_iter += 1
            
            # config.yamlに設定した頻度に合わせて検証を実施(eval_every_n_epochs=1の場合毎エポックで検証実施)
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                # 以下の_validateメソッドを呼び出し
                valid_loss = self._validate(model, valid_loader)
                print("[%d/%d] val_loss: %.3f" % (epoch_counter, self.config['epochs'], valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved')

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            # 10エポック以降はスケジューラ(学習率)を更新する
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
            # pytorch 1.4以降用にget_lrからget_last_lrに変更(参考:https://qiita.com/kurilab/items/b554744859c014f55145)
#            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

    # trainメソッドから呼び出される
    def _load_pre_trained_weights(self, model):
        # checkpoints_folder = feature_extractorディレクトリ下のrunsディレクトリ内のconfig.yamlのfine_tune_from変数で指定したディレクトリパスの中のcheckpointsディレクトリ    
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            # state_dict = checkpoints_folder内のmodel.pthをstate_dictとする。
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            # モデルの読み込み
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
    # 上記のtrainメソッドより、指定エポック毎に呼び出される
    def _validate(self, model, valid_loader):

        # validation steps
        # 勾配計算用のパラメータを保存しない(パラメータ更新しないため)
        with torch.no_grad():
            # 検証モードに切り替え(ドロップアウト層、Batch正規化層スキップ)
            model.eval()

            valid_loss = 0.0
            counter = 0
            # それぞれ異なる拡張を行った画像を2枚セットで作成している
            for (xis, xjs) in valid_loader:
                # チェック用3.29
                print(f"xis:{xis},xjs:{xjs}")
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
                # チェック用
                print(f"カウンター:{counter}")
            print(f"最終カウンター:{counter}")
            # 自作部分 (元コード：valid_loss /= counter)
            if counter == 0:
                valid_loss = valid_loss
            else:
                valid_loss /= counter
            # 自作部分は以上
        model.train()
        return valid_loss
